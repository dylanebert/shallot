# Font assets

`jetbrains-mono.ttf` is JetBrains Mono Regular, downloaded from the Google Fonts CSS `src` URL used by the cells default face:

`https://fonts.gstatic.com/s/jetbrainsmono/v24/tDbY2o-flEEny0FZhsfKu5WU4zr3E_BX0PnT8RD8yKxjPQ.ttf`

The `v24` path is the Google Fonts version captured for this asset. JetBrains Mono is distributed under the SIL Open Font License 1.1; license text and source are maintained by JetBrains at <https://github.com/JetBrains/JetBrainsMono>.

The checked-in bytes have MD5 `3d12b91dc3e06267b7eaead855a9276f`. The cells gym probes serve the same bytes as `examples/gym/public/jetbrains-mono.ttf` and assert that copy's MD5 in their source comment. `font.ttf` remains the Outfit fixture used by the text scenario and retains its original bytes.
