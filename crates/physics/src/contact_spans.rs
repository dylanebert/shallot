//! Box3D contact specs and flat prepare/store spans (contact.h and solver.h).
use crate::col::Col;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ContactSpec {
    pub contact_id: i32,
    pub manifold_start: i32,
    pub manifold_count: u16,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ContactPrepareSpan {
    pub start: i32,
    pub count: i32,
    pub contacts: *const ContactSpec,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct WidePrepareSpan {
    pub start: i32,
    pub count: i32,
    pub contacts: *const u32,
}

// SAFETY: graph arrays are read-only and cannot relocate between solver fork and join.
unsafe impl Send for ContactPrepareSpan {}
unsafe impl Send for WidePrepareSpan {}

fn span_index<T: Copy>(spans: Col<T>, start: usize, get_start: impl Fn(T) -> i32) -> usize {
    let mut i = 0;
    while i + 1 < spans.len() && get_start(spans.get(i + 1)) as usize <= start {
        i += 1;
    }
    i
}

pub(crate) fn contact_specs(
    spans: Col<'_, ContactPrepareSpan>,
    start: usize,
    count: usize,
) -> impl Iterator<Item = (usize, ContactSpec)> + use<'_> {
    let mut span = span_index(spans, start, |s| s.start);
    (start..start + count).map(move |index| {
        while spans.get(span + 1).start as usize <= index {
            span += 1;
        }
        let s = spans.get(span);
        (index, unsafe { *s.contacts.add(index - s.start as usize) })
    })
}

pub(crate) fn wide_records(
    spans: Col<'_, WidePrepareSpan>,
    start: usize,
    count: usize,
) -> impl Iterator<Item = (usize, WidePrepareSpan)> + use<'_> {
    let mut span = span_index(spans, start, |s| s.start);
    (start..start + count).map(move |index| {
        while spans.get(span + 1).start as usize <= index {
            span += 1;
        }
        (index, spans.get(span))
    })
}
