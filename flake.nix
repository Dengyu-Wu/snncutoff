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
    utils.lib.eachDefaultSystem (system: let
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

        # Enable python libraries to discover libstdc++.so.6
        LD_LIBRARY_PATH = with pkgs; lib.makeLibraryPath [ stdenv.cc.cc ];
      };
    });
}
