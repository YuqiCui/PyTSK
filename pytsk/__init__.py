import importlib.util

package_name = "torch"
spec = importlib.util.find_spec(package_name)
if spec is None:
    print("Warning! PyTorch package is not installed, "
          "this package must be installed when you need to use pytsk.gradient_descent")
