//! Chat-template rendering from the Jinja template embedded in GGUF
//! (`tokenizer.chat_template`), with a ChatML fallback.

use minijinja::{context, Environment, Error, ErrorKind};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub struct ChatTemplate {
    source: String,
    bos: String,
    eos: String,
    pub is_fallback: bool,
}

const CHATML: &str = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

impl ChatTemplate {
    pub fn new(template: Option<&str>, bos: &str, eos: &str) -> Self {
        ChatTemplate {
            source: template.unwrap_or(CHATML).to_string(),
            bos: bos.to_string(),
            eos: eos.to_string(),
            is_fallback: template.is_none(),
        }
    }

    pub fn render(&self, messages: &[ChatMessage], add_generation_prompt: bool) -> Result<String, Error> {
        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |msg: String| -> Result<String, Error> { Err(Error::new(ErrorKind::InvalidOperation, msg)) });
        env.add_function("strftime_now", |_fmt: String| -> String { String::new() });
        env.add_template("chat", &self.source)?;
        let tmpl = env.get_template("chat")?;
        tmpl.render(context! {
            messages => messages,
            add_generation_prompt => add_generation_prompt,
            bos_token => self.bos.clone(),
            eos_token => self.eos.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chatml_fallback() {
        let t = ChatTemplate::new(None, "", "");
        let s = t.render(&[ChatMessage { role: "user".into(), content: "hi".into() }], true).unwrap();
        assert_eq!(s, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");
    }

    #[test]
    fn python_methods_work() {
        let src = "{% for m in messages %}{{ m['content'].strip() }}{% if m.role.startswith('u') %}!{% endif %}{% endfor %}";
        let t = ChatTemplate::new(Some(src), "", "");
        let s = t.render(&[ChatMessage { role: "user".into(), content: "  x ".into() }], false).unwrap();
        assert_eq!(s, "x!");
    }
}
