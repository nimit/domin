# Copyright 2026 Nimit Shah. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import subprocess
import sys
import os

def test_generate_dexterous_dry_run():
    """
    Runs the examples/generate_dexterous.py script with minimal arguments
    to verify that the package imports are working and the script executes.
    Using --num_episodes 1 for a quick check.
    """
    example_script = os.path.join("examples", "generate_dexterous.py")
    
    # Needs to be run in an environment where domin is installed
    # check if file exists
    assert os.path.exists(example_script), "Example script not found"

    cmd = [
        sys.executable,
        example_script,
        "--num_episodes", "1",
        "--num_envs", "1",
        "--headless" # Assuming there is a headless flag or adding one if needed/supported by AppLauncher
    ]
    
    # We might need to handle the AppLauncher arguments. 
    # Usually AppLauncher parses sys.argv. 
    # Let's hope passing args works standardly.
    
    # Adding PYTHONPATH to include current directory if strictly needed for tests 
    # but the goal is to test installed package.
    
    print(f"Running command: {' '.join(cmd)}")
    
    # We run it and expect 0 exit code
    # We capture output to avoid spamming test logs unless failure
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        
    assert result.returncode == 0, f"Script failed with exit code {result.returncode}"
