# ISEDA in Python
This is an implementation of the inertial symmetry edge detection algorithm from
the Image Processing of Illuminated Ellipsoid paper [^1] in Python/numpy.

## Usage
You need Python and `virtualenv`.

```bash
# Install virtualenv.
python -m pip install virtualenv

# Create a virtualenv and activate it.
python -m virtualenv venv
source venv/bin/activate

# Install required packages.
pip install -r requirements.txt

# Run the algorithm.
python iseda.py
```

There is a `shell.nix` for NixOS users. Be aware that this uses `zsh` as the
runScript as that is what I use, and I haven't really bothered to make it work
on other machines.
```
nix-shell
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
python3 iseda.py
```

[^1]: Mortari, D., D’Souza, C. N., & Zanetti, R. (2016). Image Processing of
    Illuminated Ellipsoid. Journal of Spacecraft and Rockets, 53(3), 448–456.
    https://doi.org/10.2514/1.a33342
