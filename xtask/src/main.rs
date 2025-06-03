// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.

use clap::{CommandFactory, Parser, Subcommand};
use colored::Colorize;
use std::process::{Command, Stdio};

#[derive(Parser)]
#[command(name = "xtask")]
#[command(about = "Development task runner for LatchLM")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Format code
    Fmt,
    /// Run tests
    Test {
        #[arg(short, long)]
        quiet: bool,
    },
    /// Run code coverage
    Coverage,
    /// Run cargo-deny
    Deny,
    /// Run spell checks
    Spell {
        #[arg(long)]
        fix: bool,
    },
    /// Run pre-commit checks
    Check,
    /// Serve the docs
    Book,
}

const CARGO: &str = "cargo";
const CHECK_MARK: &str = "✓";
const ARROW: &str = "→";
const BULLET: &str = "⋄";

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Some(Commands::Fmt) => fmt(),
        Some(Commands::Test { quiet }) => test(quiet),
        Some(Commands::Coverage) => coverage(),
        Some(Commands::Deny) => deny(),
        Some(Commands::Spell { fix }) => spell(fix),
        Some(Commands::Check) => check(),
        Some(Commands::Book) => book(),
        None => {
            let _ = Cli::command().print_help();
            Ok("".to_string())
        }
    };

    match result {
        Ok(message) => println!("{}", message.green().bold()),
        Err(e) => println!("{}", e.to_string().red().bold()),
    }
}

fn fmt() -> Result<String, Box<dyn std::error::Error>> {
    println!("{}", "Formatting source code...".bold());

    let mut cmd = Command::new("cargo");
    cmd.args(["fmt", "--all"]);

    let status = cmd.status()?;

    if status.success() {
        Ok(format!(
            "{} {}",
            CHECK_MARK.green().bold(),
            "Formatting complete".green()
        ))
    } else {
        Err("Formatting failed".into())
    }
}

fn deny() -> Result<String, Box<dyn std::error::Error>> {
    println!(" {} {}", ARROW.blue(), "Running cargo-deny checks".bold());

    let checks = ["advisories", "bans", "licenses", "sources"];
    for check in checks {
        print!("    {} {:<12}", BULLET.blue(), check.bold());

        let output = Command::new("cargo")
            .args(["deny", "check", check])
            .output()?;

        if output.status.success() {
            println!("{}", CHECK_MARK.green());
        } else {
            println!("\n{}", "Failed!".red().bold());
            if !output.stderr.is_empty() {
                // Print the error message from cargo-deny
                println!("{}", String::from_utf8_lossy(&output.stderr));
            }
            return Err(format!("cargo-deny {check} check failed").into());
        }
    }

    Ok("Dependency checks passed".into())
}

fn spell(fix: bool) -> Result<String, Box<dyn std::error::Error>> {
    let mut cmd = Command::new("typos");

    if fix {
        cmd.arg("-w");
    }

    if !cmd.status()?.success() {
        return Err("Spell check failed. Run 'cargo xtask spell --fix' to fix.".into());
    }
    if fix {
        return Ok("Spelling fixes applied".into());
    }
    Ok("Spell check passed".into())
}

fn check() -> Result<String, Box<dyn std::error::Error>> {
    println!("\n{}", "Running checks...".bold());

    // Format check
    println!(" {} {}", ARROW.blue(), "Checking formatting".bold());
    let status = Command::new(CARGO)
        .args(["fmt", "--all", "--", "--check"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();

    if !status?.success() {
        return Err("Formatting check failed. Run 'cargo xtask fmt' to fix.".into());
    }
    println!("  {} Format check passed", CHECK_MARK.green());

    // Clippy check
    println!(" {} {}", ARROW.blue(), "Running clippy".bold());
    let status = Command::new(CARGO)
        .args([
            "clippy",
            "--all-targets",
            "--all-features",
            "--",
            "-D",
            "warnings",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();

    if !status?.success() {
        return Err("Clippy found issues. Please fix them before committing.".into());
    }
    println!("  {} Clippy check passed", CHECK_MARK.green());

    // Udeps check
    println!(
        " {} {}",
        ARROW.blue(),
        "Checking for unused dependencies".bold()
    );
    let status = Command::new(CARGO)
        .args(["+nightly", "udeps"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();

    if !status?.success() {
        return Err("Unused dependencies found".into());
    }
    println!("  {} No unused dependencies", CHECK_MARK.green());

    // Audit check
    println!(" {} {}", ARROW.blue(), "Running cargo-audit".bold());
    let status = Command::new(CARGO).args(["audit", "-q"]).status();

    if !status?.success() {
        return Err("Found a vulnerable dependency".into());
    }
    println!("  {} No vulnerable dependencies", CHECK_MARK.green());

    // Deny check
    deny()?;

    // Test
    println!(" {} {}", ARROW.blue(), "Running tests".bold());
    test(true)?;
    println!("  {} All test passed", CHECK_MARK.green());

    // Spell checks
    println!(" {} {}", ARROW.blue(), "Running spell checks".bold());
    spell(false)?;
    println!("  {} Spell checks passed", CHECK_MARK.green());

    Ok("All checks passed successfully!".to_string())
}

fn test(quiet: bool) -> Result<String, Box<dyn std::error::Error>> {
    let mut cmd = Command::new(CARGO);
    cmd.args(["nextest", "run"]);

    if quiet {
        cmd.stdout(Stdio::null());
        cmd.stderr(Stdio::null());
    }

    if !cmd.status()?.success() {
        return Err("Tests Failed".into());
    }

    Ok("Tests successfully completed".into())
}

fn coverage() -> Result<String, Box<dyn std::error::Error>> {
    let status = Command::new(CARGO)
        .args([
            "tarpaulin",
            "--all-features",
            "--skip-clean",
            "--exclude-files",
            "xtask/**",
            "--out",
            "Html",
            "--output-dir",
            "coverage",
        ])
        .stderr(Stdio::null())
        .stdout(Stdio::null())
        .status()?;

    if status.success() {
        println!(
            " {} Coverage report generated in coverage/",
            CHECK_MARK.green()
        );
        Ok("Coverage analysis complete".into())
    } else {
        Err("Failed to generate coverage report".into())
    }
}

fn book() -> Result<String, Box<dyn std::error::Error>> {
    println!(
        "{}",
        "Starting mdbook server at http://localhost:3000"
            .bold()
            .blue()
    );
    println!("{}", "Press Ctrl+C to stop the server".italic());

    if !Command::new("mdbook").arg("serve").status()?.success() {
        return Err("Failed to serve the book".into());
    }

    Ok("Book server stopped".to_string())
}
