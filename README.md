# ISEDA in Python
This is an implementation of the inertial symmetry edge detection algorithm from
the Image Processing of Illuminated Ellipsoid paper[^1] in Python/numpy.

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

# Run the algorithm on the moon sample image.
python iseda.py moon.jpg
```

### Points
There is also a `label.py` script that allows generation of "point-files" from
manually adjusted points from a GUI. This uses the ISEDA algorithm to give rough
initial points for adjustment.

A "point-file" is a `.json` file with the format of:
```jsonc
{
    "image": {
        "name": ... // name (including extension) of the image file
        "width": ... // width of the image
        "height": ... // height of the image
        "sha256": ... // sha256 digest of the image file
    },
    // list of points that are on the horizon of the celestial body in the image
    "target_points": [
        [ x_0, y_0 ], // x and y of the first point
        [ x_1, y_1 ], // x and y of the second point
        ...
    ]
}
```

### NixOS
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
