//! Explicit account setup before loading any project or granting trust.
use open_max_core::chatgpt;
use std::path::Path;

pub async fn run(dir: &Path, action: &str, account: &str) -> Result<(), String> {
    match action {
        "status" => {
            let lines = chatgpt::status(dir)?;
            if lines.is_empty() {
                println!("No ChatGPT accounts. Run openmax --chatgpt login");
            }
            for line in lines {
                println!("{line}");
            }
        }
        "logout" => println!("{}", chatgpt::logout(dir, account).await?),
        "login" => {
            let result = chatgpt::login(dir, account, |url| {
                println!("Continue with ChatGPT in your browser:\n{url}");
                #[cfg(target_os = "macos")]
                let command = "open";
                #[cfg(not(target_os = "macos"))]
                let command = "xdg-open";
                if let Ok(mut child) = std::process::Command::new(command)
                    .arg(url)
                    .stdin(std::process::Stdio::null())
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .spawn()
                {
                    std::thread::spawn(move || {
                        let _ = child.wait();
                    });
                }
            })
            .await?;
            println!("{}", result.message);
            if result.plan_enabled {
                refresh_models(dir, account).await?;
            }
        }
        "models" => refresh_models(dir, account).await?,
        _ => return Err("unknown ChatGPT operation".into()),
    }
    Ok(())
}

async fn refresh_models(dir: &Path, account: &str) -> Result<(), String> {
    let catalog = chatgpt::models(dir, account).await?;
    let models = chatgpt::save_catalog(dir, account, &catalog).await?;
    println!("Available models for chatgpt-{account}:");
    for model in &models {
        println!(
            "  {}  {}  context: {}",
            open_max_core::text::one_line(&model.id),
            open_max_core::text::one_line(model.name.as_deref().unwrap_or("")),
            model.context_tokens.map_or("not reported".into(), |n| n.to_string())
        );
    }
    println!("Select chatgpt-{account} with /model, or run openmax --provider chatgpt-{account} --model <model-id>.");
    if models.iter().any(|m| m.context_tokens.is_none()) {
        println!("For models without a reported window, supply --context-tokens from the model's documentation.");
    }
    println!("Using ChatGPT plan. Manage usage: https://chatgpt.com/settings/usage");
    Ok(())
}
