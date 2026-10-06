# Original course setup instructions

Extracted from the supplied Moodle pages before removing the saved webpages and their browser assets. Commands below reproduce the original instructions; the instructor `iml_env.yaml` has not been supplied. For this project’s reconstructed environment and browser preview, see [the proof instructions](../proof/README.md). Windows navigation screenshots were omitted.

## Running an app

For each unit, create a subfolder in your course folder (e.g., Documents/Hands_on_AI1/Unit1).

Download the app.zip from the Material tab. Unzip it and move to the unit subfolder.

Open Terminal (Anaconda Prompt for Windows)

Activate the course conda environment: conda activate iml

Go to the folder containing app.py: cd /path/to/unit_subfolder *

Launch application: shiny run --reload --launch-browser --port=0 app.py

*On Windows, you can get the /path/to/unit_subfolder by navigating the unit folder in the File Explorer and right-clicking the folder name at the top to "Copy the address as Text".

## Setting up the environment

0. Create a folder for Hands-on AI I course

For example, in Documents, create a folder Hands_on_AI1 (please avoid spaces in path names and use underscores instead). This will be referred as /path/to/your/course/folder in the following.

1. Install Miniconda

Follow the OS-specific guide: https://www.anaconda.com/docs/getting-started/miniconda/install

When finished, open a terminal:

Windows: Anaconda Prompt (miniconda3)

macOS/Linux: Terminal

Verify: conda --version

2. Create the course environment

Download iml_env.yaml to your course folder. Click the Conda Env File tab on Moodle.

In the terminal, go to that folder: cd /path/to/your/course/folder *

Create the environment (named iml): conda env create -f iml_env.yaml -n iml

Activate it: conda activate iml

*On Windows, you can get the /path/to/your/course/folder by navigating the course folder in the File Explorer and right-clicking the folder name at the top to "Copy the address as Text".

3. Install Word Embedding Model (for Unit 2)

In the activated iml environment run: pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.7.1/en_core_web_md-3.7.1-py3-none-any.whl

4. Install PyTorch (for Units 5-7)

Open https://pytorch.org/get-started/locally/

Choose:

PyTorch Build: Stable

Your OS: pick your OS

Package: pip

Language: Python

Compute Platform: pick your GPU CUDA version (or CPU if no GPU)

Copy the command shown and paste it into your activated iml environment.
