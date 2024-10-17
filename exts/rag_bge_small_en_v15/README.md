## Note on background workers

Background workers must be initialized in the extension's `_PG_init()` function and can only be started if loaded through the `shared_preload_libraries` configuration setting. So to use this extension you'll need to edit `postgresql.conf` and, depending on the platform, add:

```
shared_preload_libraries = 'rag_bge_small_en_v15.so'
```

or

 ```
shared_preload_libraries = 'rag_bge_small_en_v15.dylib'
```

When using `cargo pgrx run` with Postgres instances installed by pgrx, the `postgresql.conf` file is located in `~/.pgrx/data-N` (where N is the relevant Postgres version).

When using `cargo pgrx test`, the `postgresql.conf` file is inside the `target` directory of your extension, e.g. `~/path/to/myext/target/test-pgdata/N` (where N is the relevant Postgres version).
