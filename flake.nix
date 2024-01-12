{
  description = "SNNCutoff's development environment";

  inputs = {
    utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    utils,
    ...
  }:
    utils.lib.eachDefaultSystem (system: # FIXME: outputs for multiple systems
    let
      pkgs = import nixpkgs {
        inherit system;
      };
      pypkgs = pkgs.python3Packages;
    in {
      devShells.default = pkgs.mkShell {
        venvDir = "./.venv";
        buildInputs = [
          pypkgs.python
          pypkgs.venvShellHook  # creates a venv in $venvDir
        ];

        postVenvCreation = ''
          python -m pip install --editable .[dev]
        '';

        postShellHook = ''
        '';

        LD_LIBRARY_PATH = with pkgs; lib.makeLibraryPath [
          # Enable python libraries to discover libstdc++.so.6
          stdenv.cc.cc

          # FIXME: workaround for finding libcuda.so on non-NixOS systems
          "/usr"  # General location
          "/usr/lib/wsl"  # On WSL
        ];
      };
    });
}
