use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};

const SETUP_SQL: &str = "
    CREATE TABLE images (
        id INTEGER PRIMARY KEY,
        label TEXT NOT NULL,
        embedding F32_BLOB(4)
    );

    INSERT INTO images (id, label, embedding) VALUES
        (1, 'near_query', vector32('[0.10, 0.20, 0.30, 0.40]')),
        (2, 'also_close', vector32('[0.11, 0.19, 0.31, 0.39]')),
        (3, 'far', vector32('[0.90, 0.10, 0.10, 0.10]')),
        (4, 'opposite', vector32('[-0.10, -0.20, -0.30, -0.40]'));
";

const LIBSQL_INDEX_SQL: &str = "
    CREATE INDEX images_embedding_idx
    ON images (libsql_vector_idx(embedding));
";

const DISTANCE_SQL: &str = "
    SELECT label,
           vector_distance_cos(embedding, vector32('[0.10, 0.20, 0.30, 0.40]')) AS distance
    FROM images
    ORDER BY distance ASC
    LIMIT 4;
";

const TOP_K_SQL: &str = "
    SELECT images.id, images.label
    FROM vector_top_k('images_embedding_idx', vector32('[0.10, 0.20, 0.30, 0.40]'), 2) AS nearest
    JOIN images ON images.rowid = nearest.id;
";

#[tokio::main]
async fn main() -> Result<()> {
    let base = unique_test_dir()?;
    std::fs::create_dir_all(&base)?;

    println!("test folder: {}", base.display());

    match test_turso(&base).await {
        Ok(summary) => println!("TURSO LOCAL: OK\n{summary}"),
        Err(err) => println!("TURSO LOCAL: FAILED\n{err:#}"),
    }

    match test_libsql(&base).await {
        Ok(summary) => println!("LIBSQL LOCAL: OK\n{summary}"),
        Err(err) => println!("LIBSQL LOCAL: FAILED\n{err:#}"),
    }

    Ok(())
}

async fn test_turso(base: &Path) -> Result<String> {
    let db_path = base.join("turso-local.db");
    let db = turso::Builder::new_local(db_path.to_string_lossy().as_ref())
        .build()
        .await
        .context("failed to open local Turso database")?;
    let conn = db.connect().context("failed to connect to Turso database")?;

    conn.execute_batch(SETUP_SQL)
        .await
        .context("Turso failed setup SQL with F32_BLOB/vector32")?;
    let distances = turso_query_distances(&conn)
        .await
        .context("Turso failed vector_distance_cos query")?;

    conn.execute_batch(LIBSQL_INDEX_SQL)
        .await
        .context("Turso failed vector index creation with libsql_vector_idx")?;
    let top_k = turso_query_top_k(&conn)
        .await
        .context("Turso failed vector_top_k query")?;

    Ok(format_summary(distances, top_k))
}

async fn test_libsql(base: &Path) -> Result<String> {
    let db_path = base.join("libsql-local.db");
    let db = libsql::Builder::new_local(db_path)
        .build()
        .await
        .context("failed to open local libSQL database")?;
    let conn = db.connect().context("failed to connect to libSQL database")?;

    conn.execute_batch(SETUP_SQL)
        .await
        .context("libSQL failed setup SQL with F32_BLOB/vector32")?;
    let distances = libsql_query_distances(&conn)
        .await
        .context("libSQL failed vector_distance_cos query")?;

    conn.execute_batch(LIBSQL_INDEX_SQL)
        .await
        .context("libSQL failed vector index creation with libsql_vector_idx")?;
    let top_k = libsql_query_top_k(&conn)
        .await
        .context("libSQL failed vector_top_k query")?;

    Ok(format_summary(distances, top_k))
}

async fn turso_query_distances(conn: &turso::Connection) -> Result<Vec<(String, f64)>> {
    let mut rows = conn.query(DISTANCE_SQL, ()).await?;
    let mut out = Vec::new();
    while let Some(row) = rows.next().await? {
        out.push((row.get::<String>(0)?, row.get::<f64>(1)?));
    }
    Ok(out)
}

async fn turso_query_top_k(conn: &turso::Connection) -> Result<Vec<(i64, String)>> {
    let mut rows = conn.query(TOP_K_SQL, ()).await?;
    let mut out = Vec::new();
    while let Some(row) = rows.next().await? {
        out.push((row.get::<i64>(0)?, row.get::<String>(1)?));
    }
    Ok(out)
}

async fn libsql_query_distances(conn: &libsql::Connection) -> Result<Vec<(String, f64)>> {
    let mut rows = conn.query(DISTANCE_SQL, ()).await?;
    let mut out = Vec::new();
    while let Some(row) = rows.next().await? {
        out.push((row.get::<String>(0)?, row.get::<f64>(1)?));
    }
    Ok(out)
}

async fn libsql_query_top_k(conn: &libsql::Connection) -> Result<Vec<(i64, String)>> {
    let mut rows = conn.query(TOP_K_SQL, ()).await?;
    let mut out = Vec::new();
    while let Some(row) = rows.next().await? {
        out.push((row.get::<i64>(0)?, row.get::<String>(1)?));
    }
    Ok(out)
}

fn format_summary(distances: Vec<(String, f64)>, top_k: Vec<(i64, String)>) -> String {
    let distances = distances
        .into_iter()
        .map(|(label, distance)| format!("  {label}: distance={distance:.6}"))
        .collect::<Vec<_>>()
        .join("\n");
    let top_k = top_k
        .into_iter()
        .map(|(id, label)| format!("  {id}: {label}"))
        .collect::<Vec<_>>()
        .join("\n");

    format!("distance query:\n{distances}\nindexed top-k:\n{top_k}")
}

fn unique_test_dir() -> Result<PathBuf> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| anyhow!("system clock error: {err}"))?;
    Ok(PathBuf::from("runs").join(format!(
        "vector-test-{}",
        now.as_millis()
    )))
}
