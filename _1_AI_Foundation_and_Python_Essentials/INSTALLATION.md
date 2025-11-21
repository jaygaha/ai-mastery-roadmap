# Setting Up Your AI Development Environment (`Python`, `Anaconda`, `VS Code`)

A robust and well-organized development environment is essential to begin your AI development journey. Serving as your digital workshop, this environment offers the necessary tools, libraries, and frameworks to develop, test, and deploy AI models. A well-configured environment ensures reproducibility, efficient dependency management, and conflict-free transitions between projects.

In this lesson, we will set up a modern AI development environment using `Python`, the `Anaconda` distribution, and `Visual Studio Code` (VS Code) as the integrated development environment (`IDE`). This foundation will support all future lessons, allowing you to code with confidence and efficiency.

## The foundation of your `AI` development environment

`Python` lies at the heart of almost all modern AI and machine learning projects. Its readability, extensive libraries, and large user base have made it the de facto language for data scientists and AI engineers. However, managing different projects' `Python` versions, packages, and dependencies can quickly become complex. This is where tools like Anaconda come into play, offering a streamlined solution for scientific computing. An Integrated Development Environment (`IDE`) like `Visual Studio Code` brings all these components together into a productive workflow by providing an intuitive interface for writing code, debugging, and project management.

### Why Python for AI and Machine Learning?

`Python's` rise to prominence in AI and ML is not accidental. Several key features contribute to its widespread adoption:

- **Simplicity and Readability**: Python's clear syntax makes it easy to learn, write, and understand, which speeds up development time and collaboration. This is crucial for experimenting with complex algorithms.
- **Vast Ecosystem of Libraries**: Python boasts an unparalleled collection of libraries specifically designed for AI, ML, and data science. Libraries like NumPy for numerical operations, Pandas for data manipulation, Scikit-learn for traditional machine learning algorithms, and TensorFlow/PyTorch for deep learning provide powerful, optimized functionalities.
- **Large and Active Community**: A massive global community means abundant resources, tutorials, and support forums. When you encounter a problem, chances are someone else has faced it and found a solution.
- **Platform Independence**: Python code can run on various operating systems (Windows, macOS, Linux) with minimal to no modification, making it highly versatile for different deployment scenarios.

## Introducing `Anaconda`: Data Science Hub

While you could install Python directly, Anaconda offers a superior experience for data science and AI development. `Anaconda` is more than just a Python installer; it's a comprehensive data science platform that includes:

- **Conda Package Manager**: A powerful, language-agnostic package manager that can install, update, and manage software packages and their dependencies. Unlike pip (Python's native package manager), conda can manage packages that are not Python-specific, such as R packages or system libraries.
- **Environment Manager**: This is one of Anaconda's most crucial features. It allows you to create isolated environments, each with its own Python version and set of installed libraries. This prevents conflicts between projects requiring different versions of the same library. For instance, Project A might need TensorFlow==2.5.0 while Project B requires TensorFlow==2.10.0. Without environment management, installing one would break the other. Conda environments ensure these projects can coexist peacefully.
- **Bundled Libraries**: Anaconda comes pre-installed with over 250 popular data science packages (including NumPy, Pandas, and Scikit-learn) and automatically installs more than 7,500 additional packages from the Anaconda repository. This "batteries-included" approach saves significant time and effort in setting up a new project.
- **Anaconda Navigator**: A desktop graphical user interface (GUI) that allows you to launch applications and manage conda packages, environments, and channels without using command-line commands.

Imagine you're developing a machine learning model for predicting customer churn for an e-commerce platform. This project requires specific versions of data processing libraries and a particular machine learning framework. At the same time, you might be working on another project—such as image classification—that uses a different set of deep learning libraries with conflicting dependency requirements. Anaconda's environment manager allows you to create separate, independent workspaces for each project, ensuring that the libraries and their versions for your churn prediction model do not interfere with your image classification project, and vice versa. This isolation is critical for maintaining project integrity and reproducibility.

## Visual Studio Code: The Modern AI Developer's IDE

Visual Studio Code (`VS Code`) is a free, open-source, and highly popular code editor developed by Microsoft. While technically a code editor, its extensive capabilities—thanks to a rich ecosystem of extensions—often blur the lines with full-fledged IDEs. For AI development, VS Code offers:

- **IntelliSense**: Smart code completion, parameter info, and quick info for Python modules and methods, significantly speeding up coding and reducing errors.
- **Integrated Debugging**: Step through your code, inspect variables, and set breakpoints directly within the editor—an invaluable feature for identifying and fixing issues in your AI models.
- **Built-in Git Integration**: Seamlessly manage your code versions and collaborate with others using Git and platforms like GitHub.
- **Extensibility**: A marketplace with thousands of extensions for language support, linting, formatting, debugging, cloud integration, and more. The Python extension, in particular, is robust, offering features tailored for Python development.
- **Terminal Integration**: Run commands directly within VS Code, making it easy to manage your conda environments, install packages, and execute Python scripts without switching applications.

For instance, when developing the customer churn prediction model, you'll be writing Python code to preprocess data, define model architectures, and train the models. VS Code will highlight syntax errors in real time, suggest completions for library functions (e.g., pd.DataFrame(...)), and allow you to set a breakpoint in your training loop to inspect how your model's weights are changing or if your data transformations are working as expected. This immediate feedback and integrated toolset significantly enhance productivity compared to using a basic text editor.

## Setting Up Environment: Step-by-Step Implementation

The following is a step-by-step walkthrough for setting up `Python`, `Anaconda`, and `VS Code`.

### 1. Installing Anaconda

The first step is to download and install the Anaconda Distribution.

1. Download Anaconda:
   1. Go to the official Anaconda website: [https://www.anaconda.com/products/individual](https://www.anaconda.com/products/individual)
   2. Download the appropriate graphical installer for your operating system (Windows, macOS, or Linux). Ensure you select the `Python` `3.x` version (as Python 2.x is deprecated and no longer supported).
2. Install Anaconda:
   1. **Windows**
      1. Run the `.exe` installer and follow the prompts.
      2. **Installation Scope:** Select **"Just Me"** unless multi-user requirements exist.
      3. **PATH Environment Variable:** Check **"Add Anaconda to my PATH environment variable"** (recommended for beginners to simplify command line usage).
      4. **Default Python:** Register Anaconda as your default **Python 3.x**.
   2. **macOS**
      1. Run the `.pkg` installer and follow the prompts.
      2. **Install Location:** Choose **"Install for me only"**.
   3. **Linux**
      1. Open a terminal and navigate to the directory with the `.sh` file. 
      2. Run:
         ```bash
         bash Anaconda3-*-Linux-x86_64.sh
         ```
      3. Follow the prompts:
         1. Press **Enter** to read the license.
         2. Type **yes** to accept the license.
         3. Accept the default installation location (unless you have a specific reason to change it).
         4. Type **yes** to initialize Anaconda3 by running `conda init`.
3. Verify Installation:
   1. Open a new terminal (or Anaconda Prompt on Windows).
   2. Type `conda --version` and press Enter. You should see the conda version number.
   3. Type `python --version` and press Enter. You should see the Python version that came with Anaconda.
   4. Type `conda list` and press Enter. This will show all packages pre-installed with your base Anaconda environment.

### 2. Creating and Managing Conda Environments

Now that Anaconda is installed, let's create a dedicated environment for our AI development course.

1. Open Terminal/Anaconda Prompt:
   1. **Windows**: Search for "Anaconda Prompt" in your Start Menu and open it.
   2. **macOS/Linux**: Open your regular terminal application.
2. Create a New Conda Environment:
   1. We will name our environment `ai_course_env` (generally `.venv`).
   2. Run the command:`````
      ```bash
      conda create -n ai_course_env python=3.13
      ```
      1. `conda create`: The command to create a new environment. 
      2. `-n ai_course_env`: Specifies the name of the new environment as `ai_course_env`.
      3. `python=3.13`: Specifies that this environment should use Python version 3.13. You can choose a different Python version if needed, but 3.13 is a good modern choice.
   3. When prompted to proceed (`[y/N]`), type `y` and press Enter. Conda will download and install the necessary Python version and basic packages into this new, isolated environment.
3. Activate the Environment:
   1. Before you can use an environment, you must activate it.
   2. Run the command:
      ```bash
      conda activate ai_course_env
      ```
   3. You'll notice your terminal prompt changes, usually prefixing the current environment's name (e.g., (`ai_course_env`) `C:\Users\YourUser>`, `(ai_course_env) gaha/ai_course_roadmap`). This indicates that you are now working within `ai_course_env`.
   4. Install Essential Packages:
      1. Once activated, you can install specific libraries needed for your projects. For this course, we will initially install `numpy` and `pandas`, which are fundamental for data manipulation, and `scikit-learn` for traditional machine learning. These will be used in upcoming lessons.
      2. Run the command
         ```bash
         conda install numpy pandas scikit-learn matplotlib
         ```
         1. `conda install`: The command to install packages into the active environment.
         2. `numpy pandas scikit-learn matplotlib`: The names of the packages to install. matplotlib is included for basic plotting.
      3. Type `y` when prompted to proceed.
   5. Verify Installed Packages:
      1. To see what packages are installed in your _active_ environment: 
         ```bash
         conda list
         ``` 
         You should now see `numpy`, `pandas`, `scikit-learn`, and `matplotlib` along with their dependencies.
   6. Deactivate the Environment (Optional): 
      1. When you are done working in a specific environment, you can deactivate it.
      2. Run the command: 
         ```bash
         conda deactivate
         ```
         Your terminal prompt will revert to its previous state (e.g., `(base)` or no prefix).

### 3. Installing Visual Studio Code

Next, we will install Visual Studio Code.

1.  Download VS Code:
    1. Go to the official VS Code website: [https://code.visualstudio.com/](https://code.visualstudio.com/)
    2. Download the stable build for your operating system (Windows, macOS, or Linux).
2. Install VS Code:
   1. **Windows**: Run the `.exe` installer. Accept the license agreement, choose the installation location, and click "Next". On the "Select Additional Tasks" screen, it's highly recommended to check "Add "Open with Code" action to Windows Explorer file context menu" and "Register Code as an editor for supported file types" for convenience. 
   2. **macOS**: Open the downloaded `.zip` file, which will extract the "Visual Studio Code" application. Drag this application into your Applications folder. 
   3. **Linux**: Depending on your distribution, you might use `snap install --classic code` or` sudo apt install code` after adding the repository. Follow instructions specific to your distribution.
3. Verify Installation:
   1. Open VS Code. You should see its welcome screen.

### 4. Configuring Visual Studio Code for Python and Conda

Finally, we integrate VS Code with our Conda environment.

1. Install the Python Extension:
   1. Open VS Code. 
   2. Go to the Extensions view by clicking the square icon on the left sidebar or pressing `Ctrl+Shift+X` (Windows/Linux) or `Cmd+Shift+X` (macOS). 
   3. Search for "Python" by Microsoft. 
   4. Click "Install". This extension provides rich support for Python development, including IntelliSense, linting, debugging, and environment selection.
2. Select Your Conda Environment as the Python Interpreter:
   1. Open VS Code.
   2. Open the Command Palette by pressing `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS). 
   3. Type `Python: Select Interpreter` and select this option from the dropdown. 
   4. VS Code will display a list of detected Python interpreters. Look for the one associated with your `ai_course_env` (it might show up as conda (`ai_course_env`) or similar). Select it. 
   5. You should see the selected interpreter displayed in the bottom-left corner of the VS Code window (e.g., `Python 3.13.x (conda) 'ai_course_env'`).

### 5. Writing and Running Your First Python Script in VS Code

Let's test our setup by writing a simple Python script.

1. Create a New File:
   1. In VS Code, go to `File > New Text File` or press `Ctrl+N` (Windows/Linux) / `Cmd+N` (macOS). 
   2. Save the file immediately as `hello_ai.py` (e.g., in a new folder named `ai_course_projects` on your desktop). Make sure the file extension is `.py` so VS Code recognizes it as a Python file.
2. Write Simple Python Code:
   1. Type the following code into `hello_ai.py`:
   ```python
    # This is our first Python script in the AI development environment!

    # Define a simple variable
    message = "Hello, AI Development Roadmap!"
    
    # Print the message to the console
    print(message)
    
    # Demonstrate a simple arithmetic operation
    x = 5
    y = 10
    sum_result = x + y
    print(f"The sum of {x} and {y} is: {sum_result}")
    
    # Import a library installed in our conda environment (NumPy for example)
    # We will learn more about NumPy in a future lesson.
    import numpy as np
    
    # Create a simple NumPy array
    arr = np.array([1, 2, 3, 4, 5])
    print(f"A NumPy array: {arr}")
   ```
3. Run the Script:
   1. With `hello_ai.py` open in VS Code, you have several ways to run it:
      1. **Click the Run Button**: In the top-right corner of the editor, there's often a small "Play" triangle icon. Click it to run the file. 
      2. **Right-Click in Editor**: Right-click anywhere in the code editor and select "Run Python File in Terminal". 
      3. **Using the Terminal**: Open the integrated terminal in VS Code (`Ctrl+` or `Cmd+`). Ensure your `ai_course_env` is active (you should see `(ai_course_env)` at the prompt). Then, navigate to the directory where you saved `hello_ai.py` using `cd` commands (e.g., `cd Desktop\ai_course_projects`). Finally, run the script using: `python hello_ai.py`
   2. The output of your script will appear in the integrated terminal within VS Code. You should see:
      ```bash
      Hello, AI Development Roadmap!
      The sum of 5 and 10 is: 15
      A NumPy array: [1 2 3 4 5]
      ```
          
      If you see this output, congratulations! Your AI development environment is successfully set up.

## Conclusion

**AI Development Setup Complete**

You’ve built a **professional AI workspace** using:
- **Python** (core language)
- **Anaconda** (package/environment management)
- **VS Code** (IDE)

**Why it matters:**
- Ensures a **stable, efficient, and reproducible** environment.
- Avoids dependency conflicts, letting you work on **multiple projects smoothly**.

**What’s next:**
- Dive into **Python basics** (data structures, control flow)—the foundation for your AI projects. [Go to the next step ➡️](crash-course/README.md)