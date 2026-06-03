"""
Orchestration script to sequentially execute the pipeline tasks: parameter
configuration, core algorithm training, and feedback loop validation.
"""

import subprocess
import sys

def run_script(script_name: str):
    """
    Executes an external Python script as a subprocess and streams output to stdout.
    Terminates the orchestration pipeline immediately if a script returns a non-zero exit code.
    """
    print("\n" + "="*60)
    
    # Run the script and pipe execution output in real time to the current terminal
    result = subprocess.run([sys.executable, script_name], text=True)
        
    print(f"--> Completed {script_name}")


if __name__ == "__main__":
    # Ordered list mapping structural pipeline execution stages
    scripts = [
        "param_configuration.py",
        "rvi_training.py",
        "validation_lcc_rvi.py"
    ]
        
    for script in scripts:
        run_script(script)
        
    print("\n" + "="*60)