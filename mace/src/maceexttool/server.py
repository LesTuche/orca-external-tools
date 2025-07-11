from __future__ import annotations

import logging
import sys
import threading
from typing import Callable
from ase import Atoms

import torch
import waitress
from flask import Flask, request, jsonify

from maceexttool import common, calculator

app = Flask('maceserver')

model: str = ''  # will hold the selected model

# will hold one MACE Calculator per server thread
calculators: dict[int, Callable] = {}


@app.route('/calculate', methods=['POST'])
def run_mace():
    """
    Runs a MACE calculation.
    Expects a JSON payload, which can be deserialized directly as kwargs to MACE Calculator, i.e.:
    {
        "data": {
            "coord": list[list[tuple[float, float, float]]],
            "numbers": list[list[int]],
            "charge": list[list[float]],
            "mult": list[list[int]],
        },
        "forces": bool (optional),
        "stress": currently not possible with MACE,
        "hessian": currently not possible with MACE,
        "nthreads": int  # passed to torch.set_num_threads()
    }
    """
    # Save input from client (is a JSON file)
    input = request.get_json()

    # Make ASE atoms object and add variables sent from client
    atoms = Atoms(symbols=input["atom_types"], positions=input["coordinates"])
    atoms.info = {"charge": input["charge"], "spin": input["mult"]}

    # Set the number of torch threads
    thread_id = threading.get_ident()
    # Get the initialized MACE Calculator
    # Since the object is not thread-safe, we initialize one per server thread
    thread_id = threading.get_ident()
    global calculators
    if thread_id not in calculators:
        calculators[thread_id] = calculator.init(model)
    calc = calculators[thread_id]

    # run the calculation
    atoms.calc = calc

    # get the output
    energy, gradient = common.process_output(atoms)

    return jsonify({'energy': energy, 'gradient': gradient})


def run(arglist: list[str]):
    """Start the MACE calculation server using a specified model file."""
    args = common.cli_parse(arglist, mode=common.RunMode.Server)

    # get the absolute path of the model file as a plain string
    global model
    model = str(args.model)

    # set up logging
    logger = logging.getLogger('waitress')
    logger.setLevel(logging.DEBUG)

    # start the server
    waitress.serve(app, listen=args.bind, threads=args.nthreads,channel_timeout=600,cleanup_interval=5,connection_limit=1000)


def main():
    """Entry point for CLI execution"""
    run(sys.argv[1:])


if __name__ == '__main__':
    main()
