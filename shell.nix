let
    pkgs = import <nixpkgs> {};
    #optix = pkgs.fetchzip {
    #  url = "https://developer.download.nvidia.com/redist/optix/v7.3/OptiX-7.3.0-Include.zip";
    #  hash = "sha256-aMrp0Uff4c3ICRn4S6zedf6Q4Mc0/duBhKwKgYgMXVU=";
    #};
in
pkgs.mkShell {
    packages = with pkgs; [
        cmake
        pkg-config
        xorg.libX11
        xorg.libXrandr
        xorg.libXinerama
        xorg.libXcursor
        xorg.libXxf86vm
        xorg.libXi
        ocl-icd
        opencl-headers
        glfw
        libdevil
        # Optix (need `export NIXPKGS_ALLOW_UNFREE=1 && nix-shell --impure shell-linux.nix`):
        # cudatoolkit
        # optix
    ];
}