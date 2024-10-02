use pgrx::prelude::*;

#[pg_schema]
mod rag {
    use super::super::errors::*;
    use pgrx::prelude::*;
    use std::io::Cursor;

    #[pg_extern(immutable, strict)]
    pub fn text_from_docx(document: Vec<u8>) -> String {
        // DocxError doesn't implement Display, so we can't use .expect_or_pg_err()
        let file = match docx_rust::DocxFile::from_reader(Cursor::new(document)) {
            Err(err) => error!("{ERR_PREFIX} Couldn't read .docx file: {:?}", err),
            Ok(value) => value,
        };
        let docx = match file.parse() {
            Err(err) => error!("{ERR_PREFIX} Couldn't parse .docx file: {:?}", err),
            Ok(value) => value,
        };
        docx.document.body.text()
    }
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use super::rag::*;
    use pgrx::prelude::*;

    #[pg_test]
    fn test_text_from_docx() {
        assert_eq!(
            text_from_docx(include_bytes!("../test_res/test.docx").to_vec()),
            "Test .docx document\r\n\r\n\r\nLorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.\r\n\r\n\r\n\r\nOne\r\nTwo\r\nThree\r\n\r\nDuis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
        )
    }

    #[pg_test(error = "[rag] Couldn't read .docx file: Zip(FileNotFound)")]
    fn test_text_from_not_docx() {
        text_from_docx(include_bytes!("../test_res/test.pages").to_vec());
    }
}
