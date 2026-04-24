#!/usr/bin/env python3
"""
Wrapper for g-xTB via the xtb binary (github.com/grimme-lab/g-xtb), compatible with ORCA's ExtTool interface.
g-xTB is activated by passing --gxtb to the xtb binary (v6.7.1+). No external parameter files are required.

Provides
--------
class: GxtbCalc(BaseCalc)
    Class for performing a g-xTB calculation together with ORCA
main: function
    Main function
"""

import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from oet.core.base_calc import BaseCalc, CalculationData
from oet.core.misc import (
    check_file,
    check_path,
    mult_to_nue,
    nat_from_xyzfile,
    run_command,
    write_to_file,
)


class GxtbCalc(BaseCalc):
    @property
    def PROGRAM_NAMES(self) -> list[str]:
        """Program names to search for in PATH"""
        return ["xtb"]

    @classmethod
    def extend_parser(cls, parser: ArgumentParser) -> None:
        """Add gxtb parsing options.

        Parameters
        ----------
        parser: ArgumentParser
            Parser that should be extended
        """
        parser.add_argument("-x", "--exe", dest="prog", help="Path to the xtb executable (g-xTB build)")

    def run_gxtb(
        self,
        calc_data: CalculationData,
        args: list[str],
    ) -> None:
        """
        Run the xtb program with --gxtb and redirect its STDOUT and STDERR to a file.

        Parameters
        ----------
        calc_data: CalculationData
            Settings for the calculation
        args : list[str, ...]
            additional arguments to pass to xtb
        """

        old_omp = os.environ.get("OMP_NUM_THREADS", "<unset>")
        os.environ["OMP_NUM_THREADS"] = str(calc_data.ncores)
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
        print(f"[DEBUG gxtb] OMP_NUM_THREADS: {old_omp!r} -> {os.environ['OMP_NUM_THREADS']!r}")
        print(f"[DEBUG gxtb] MKL_NUM_THREADS=1  OMP_MAX_ACTIVE_LEVELS=1")

        # xyzfile is positional in xtb; --gxtb activates the g-xTB method
        args = [calc_data.xyzfile.name, "--gxtb", "-P", str(calc_data.ncores)] + args

        if calc_data.dograd:
            args += ["--grad"]

        print(f"[DEBUG gxtb] CWD={os.getcwd()}")
        print(f"[DEBUG gxtb] cmd={calc_data.prog_path} {args}")

        if not calc_data.prog_path:
            raise RuntimeError("Path to program is None.")
        run_command(calc_data.prog_path, calc_data.output_file, args)
        print(f"[DEBUG gxtb] xtb finished OK")

        return

    def read_gxtbout(
        self, stdout_out: str | Path, grad_out: str | Path, natoms: int, dograd: bool
    ) -> tuple[float, list[float]]:
        """
        Read the output from gxtb

        Parameters
        ----------
        stdout_out: str | Path
            xtb stdout log file
        grad_out: str | Path
            file with gradient
        natoms: int
            number of atoms in the system
        dograd: bool
            whether to read the gradient

        Returns
        -------
        float
            The computed energy
        list[float]
            The gradient (X,Y,Z) for each atom
        """
        energy = None
        gradient = []
        # read energy from stdout (always present regardless of --grad)
        stdout_path = check_path(stdout_out)
        with stdout_path.open() as f:
            for line in f:
                if "TOTAL ENERGY" in line:
                    energy = float(line.split()[3])
                    break

        if not energy:
            raise ValueError("Energy couldn't be found in gxtb output.")
        # read the gradient from the turbomole gradient file (only written when --grad passed)
        if dograd:
            grad_path = check_path(grad_out)
            natoms_read = 0
            with grad_path.open() as f:
                for line in f:
                    if "$grad" in line:
                        break
                for line in f:
                    fields = line.split()
                    if len(fields) == 4:
                        natoms_read += 1
                    elif len(fields) == 3:
                        gradient += [float(i.replace("D", "E")) for i in fields]
                    elif "$end" in line:
                        break
                if natoms_read != natoms:
                    print(
                        f"Number of atoms read: {natoms_read} does not match the expected: {natoms}"
                    )
                    sys.exit(1)
                if len(gradient) != 3 * natoms:
                    print(
                        f"Number of gradient entries: {len(gradient)} does not match 3x number of atoms: {natoms}"
                    )
                    sys.exit(1)

        return energy, gradient

    def calc(
        self,
        calc_data: CalculationData,
        args_parsed: dict[str, Any],
        args_not_parsed: list[str],
    ) -> tuple[float, list[float]]:
        """
        Routine for calculating energy and optional gradient.
        Writes ORCA output

        Parameters
        ----------
        calc_data: CalculationData
            Parameters of the calculation
        args_parsed: dict[str, Any]
            Arguments parsed as defined in extend_parser
        args_not_parsed: list[str]
            Arguments not parser so far

        Returns
        -------
        float
            The computed energy (Eh)
        list[float]
            Flattened gradient vector (Eh/Bohr), if computed, otherwise empty.
        """
        # Get the arguments parsed as defined in extend_parser
        prog = args_parsed.get("prog")
        calc_data.set_program_path(prog)
        # Set and check the program path if its executable
        calc_data.set_program_path(prog)
        if calc_data.prog_path:
            print(f"Using executable {calc_data.prog_path}")
        else:
            raise FileNotFoundError(
                f"Could not find a valid executable from standard program names: {self.PROGRAM_NAMES}"
            )

        print(f"[DEBUG gxtb] calc() start — CWD={os.getcwd()}")
        print(f"[DEBUG gxtb] tmp_dir={calc_data.tmp_dir}  orca_input_dir={calc_data.orca_input_dir}")
        print(f"[DEBUG gxtb] xyzfile={calc_data.xyzfile}  ncores={calc_data.ncores}  dograd={calc_data.dograd}")

        # write .CHRG and .UHF file
        write_to_file(content=calc_data.charge, file=".CHRG")
        write_to_file(content=mult_to_nue(calc_data.mult), file=".UHF")

        # run gxtb
        self.run_gxtb(calc_data=calc_data, args=args_not_parsed)

        # get the number of atoms from the xyz file
        natoms = nat_from_xyzfile(xyz_file=calc_data.xyzfile)

        gradient_out = "gradient"

        # parse the gxtb output
        print(f"[DEBUG gxtb] reading output from {calc_data.output_file}  gradient_out={gradient_out}")
        energy, gradient = self.read_gxtbout(
            stdout_out=calc_data.output_file,
            grad_out=gradient_out,
            natoms=natoms,
            dograd=calc_data.dograd,
        )
        print(f"[DEBUG gxtb] calc() done — energy={energy:.10f}  |grad|={len(gradient)}")

        return energy, gradient


def main() -> None:
    """
    Main routine for execution
    """
    calculator = GxtbCalc()
    inputfile, args, args_not_parsed = calculator.parse_args()
    calculator.run(inputfile=inputfile, args_parsed=args, args_not_parsed=args_not_parsed)


# Python entry point
if __name__ == "__main__":
    main()
