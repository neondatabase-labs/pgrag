use pgrx::prelude::*;

#[pg_schema]
mod rag {
    use super::super::errors::*;
    use pgrx::prelude::*;

    #[pg_extern(immutable, strict)]
    pub fn text_from_pdf(document: &[u8]) -> String {
        pdf_extract::extract_text_from_mem(&document).expect_or_pg_err("Error extracting text from PDF")
    }
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use super::rag::*;
    use pgrx::prelude::*;

    #[pg_test]
    fn test_text_from_pdf() {
        assert_eq!(
            text_from_pdf(include_bytes!("../test_res/test.pdf")),
            "\n\nTest PDF document\n\n\nLorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut \nlabore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco \nlaboris nisi ut aliquip ex ea commodo consequat.\n\n\n• One\n\n• Two\n\n• Three\n\n\nDuis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. \nExcepteur sint occaecat cupidatat non proident, sunt in culpa qui oﬃcia deserunt mollit anim id \nest laborum."
        )
    }

    #[pg_test(error = "[rag] Error extracting text from PDF: PDF error: Invalid file header")]
    fn test_text_from_not_pdf() {
        text_from_pdf(include_bytes!("../test_res/test.pages"));
    }
}