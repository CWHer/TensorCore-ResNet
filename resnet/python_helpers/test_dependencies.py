#!env python

# This is used to test whether Python have the correct dependencies installed.

def dependency_check(package_name):
    try:
        __import__(package_name)
    except ImportError:
        raise ImportError(f"Package {package_name} is not installed")


dependency_names = ["argparse", "numpy", "torch", "tqdm", "PIL", "torch", "torchvision"]

if __name__ == "__main__":
    for dependency_name in dependency_names:
        dependency_check(dependency_name)
