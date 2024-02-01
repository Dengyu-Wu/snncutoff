{
  description = "SNNCutoff's development environment";

  inputs = {
    utils.url = "github:numtide/flake-utils";

    # TODO: CUDA should work without explicitly invoking a wrapper
    nixglhost.url = "github:numtide/nix-gl-host";
  };

  outputs = {
    self,
    nixpkgs,
    utils,
    nixglhost,
    ...
  }:
    utils.lib.eachDefaultSystem (system: # TODO: deprecate flake-utils
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
          nixglhost.defaultPackage.${system}
        ];

        postVenvCreation = ''
          python -m pip install --upgrade pip
          python -m pip install --editable .[dev]
        '';

        postShellHook = ''
        '';

        # Enable python libraries to discover libstdc++.so.6
        LD_LIBRARY_PATH = with pkgs; lib.makeLibraryPath [ stdenv.cc.cc ];
      };
    });
}
