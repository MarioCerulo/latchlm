{ pkgs, lib, config, inputs, ... }:

{
  # Automatically load `.env` files
  dotenv.enable = true;

  git-hooks.enable = false;

  # Enable rust tooling
  languages.rust.enable = true;
  languages.rust.channel = "nightly";

  packages = [
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
}
