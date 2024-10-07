use pgrx::prelude::*;

#[pg_schema]
mod rag {
    use super::super::errors::*;
    use pgrx::prelude::*;
    use htmd::*;

    #[pg_extern(immutable, strict)]
    pub fn markdown_from_html(document: &str) -> String {
        let converter = HtmlToMarkdown::builder().skip_tags(vec!["head", "script", "style"]).build();
        converter.convert(document).expect_or_pg_err("Error converting HTML to Markdown")
    }
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use super::rag::*;
    use pgrx::prelude::*;

    #[pg_test]
    fn test_markdown_from_html() {
        assert_eq!(
            markdown_from_html("<html><script>alert();</script><body><h1>Heading</h1><p>Para</p></body></html>"),
            "# Heading\n\nPara"
        )
    }
}
