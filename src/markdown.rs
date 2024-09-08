use pgrx::prelude::*;

#[pg_schema]
mod neon_ai {
    use super::super::errors::*;
    use pgrx::prelude::*;

    #[pg_extern(immutable, strict)]
    pub fn markdown_from_html(document: &str) -> String {
        htmd::convert(document).expect_or_pg_err("Error converting HTML to Markdown")
    }
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;
    use super::neon_ai::*;

    #[pg_test]
    fn test_markdown_from_html() {
        assert_eq!(
            markdown_from_html("<html><body><h1>Heading</h1><p>Para</p></body></html>"),
            "# Heading\n\nPara"
        )
    }
}
