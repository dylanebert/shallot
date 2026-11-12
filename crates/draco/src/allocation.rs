use crate::Result;

pub(crate) fn push<T>(values: &mut Vec<T>, value: T) -> Result<()> {
    if values.len() == values.capacity() {
        values
            .try_reserve(1)
            .map_err(|_| "Draco allocation failed")?;
    }
    values.push(value);
    Ok(())
}

pub(crate) fn copy(bytes: &[u8]) -> Result<Vec<u8>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(bytes.len())
        .map_err(|_| "Draco allocation failed")?;
    values.extend_from_slice(bytes);
    Ok(values)
}

pub(crate) fn filled<T: Clone>(count: usize, value: T) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| "Draco allocation failed")?;
    values.resize(count, value);
    Ok(values)
}
