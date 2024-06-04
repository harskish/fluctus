let
    pkgs = import <nixpkgs> {};
in
pkgs.mkShell {
    packages = with pkgs.darwin.apple_sdk.frameworks; [
        pkgs.libdevil
        pkgs.glfw
        pkgs.cmake
        pkgs.pkg-config
        Cocoa
        Kernel
    ];
}