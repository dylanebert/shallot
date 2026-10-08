//! Oracle-only observation of task versus orchestrator work.
#[derive(Clone, Copy)]
pub(crate) enum Work {
    PairParallel,
    PairSerial,
    ContactParallel,
    ContactSerial,
    ContinuousParallel,
    ContinuousSerial,
}
#[cfg(feature = "box3d-oracle")]
static COUNTS: [core::sync::atomic::AtomicUsize; crate::regions::MAX_WORLDS * 6] =
    [const { core::sync::atomic::AtomicUsize::new(0) }; crate::regions::MAX_WORLDS * 6];
#[inline(always)]
pub(crate) fn note(world: usize, work: Work) {
    #[cfg(feature = "box3d-oracle")]
    COUNTS[world * 6 + work as usize].fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    #[cfg(not(feature = "box3d-oracle"))]
    let _ = (world, work);
}
#[inline(always)]
pub(crate) fn reset(world: usize) {
    #[cfg(feature = "box3d-oracle")]
    for count in &COUNTS[world * 6..world * 6 + 6] {
        count.store(0, core::sync::atomic::Ordering::Relaxed);
    }
    #[cfg(not(feature = "box3d-oracle"))]
    let _ = world;
}
#[cfg(feature = "box3d-oracle")]
#[export_name = "box3dCallbackWork"]
pub extern "C" fn count(world: usize, lane: usize) -> usize {
    COUNTS[world * 6 + lane].load(core::sync::atomic::Ordering::Relaxed)
}
