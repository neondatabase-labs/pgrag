#[macro_export]
macro_rules! mconst {
  ($name:ident, $value:literal) => {
      macro_rules! $name {
          () => {
              $value
          };
      }
  }; 
}
