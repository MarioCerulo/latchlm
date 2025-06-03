{
  description = "Rust project development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            # Core Development & CI Tools
            pkgs.cargo-nextest # Faster test runner
            pkgs.cargo-tarpaulin # Code coverage
            pkgs.mdbook

            # Dependency & Security Auditing
            pkgs.cargo-audit # Audit dependencies for security vulnerabilities
            pkgs.cargo-deny # Check licenses, bans, duplicates, advisories

            # Development Workflow Helpers
            pkgs.cargo-edit # Add/remove/upgrade dependencies via CLI (cargo add/rm/upgrade)
            pkgs.cargo-workspaces # Manage workspaces
            pkgs.cargo-udeps # Find unused dependencies
            pkgs.bacon # Background code checker
            pkgs.typos # Spellchecker
          ];
        };
      }
    );
}
