import Pkg
Pkg.activate("./")
Pkg.instantiate()  # Ensure all dependencies listed in Project.toml are installed
Pkg.resolve()      # Ensure the package dependency graph is consistent
Pkg.update()       # Fetch the latest versions of your packages