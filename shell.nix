{
  pkgs ? import <nixpkgs> { }
}: 

(pkgs.buildFHSEnv {
  name = "pippy";
  targetPkgs = pkgs: (with pkgs; [
    python3
    python313Packages.pip
    python313Packages.virtualenv

    # the following are required for the TkAgg matplotlib backend
    python313Packages.tkinter
    python313Packages.pygobject3
    python313Packages.ipython
    python313Packages.pyqt6
    gtk3
    gobject-introspection
    librsvg
  ]);
  runScript = "zsh";
  profile = ''
    if ! python -c 'import tkinter' 2>/dev/null >/dev/null;then
      # Fix tkinter import (was necessary at some point?)
      for d in ${toString pkgs.python3Packages.tkinter}/lib/python*/site-packages;do
        export PYTHONPATH="$d:$PYTHONPATH"
      done
      if ! python -c 'import tkinter' 2>/dev/null >/dev/null;then
        echo "Still not possible to 'import tkinter', despite setting PYTHONPATH=$PYTHONPATH"
      fi
    fi
    export MPLBACKEND=TkAgg # force the TkAgg backend as that seems to be the only one that works?
  '';
}).env
