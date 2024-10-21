macro_rules! mconst {
  ($name:ident, $value:literal) => {
      macro_rules! $name {
          () => {
              $value
          };
      }
  }; 
}

pub fn retrieve_panic_message(panic: &Box<dyn std::any::Any + Send>) -> Option<&str> {
    panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&'static str>().map(std::ops::Deref::deref))
}
