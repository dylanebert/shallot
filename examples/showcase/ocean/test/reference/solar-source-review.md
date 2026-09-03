# Solar disk source review

The demo uses a mean apparent solar angular radius of `0.00465 rad` (`0.266°`, half of the familiar roughly `0.53°` diameter) and the continuum limb profile

`I(μ) / I(1) = μ^0.5`, where `μ = sqrt(1 - (r/R)^2)`.

The limb law is equation (6)'s power-law representation in Hestroffer & Magnan, “Wavelength dependency of the Solar limb darkening,” *Astronomy and Astrophysics* 333 (1998), 338–342: <https://articles.adsabs.harvard.edu/pdf/1998A%26A...333..338H>. The retrieved five-page source PDF had SHA-256 `93ba22418ff3726a7b513def68ab9aef3bebd12a0b2863a85ac360a038b38ba6`; its visible-continuum fit motivates the bounded `α = 0.5` profile used here.

The angular radius is the small-angle result from the IAU 2015 nominal solar radius `6.957e8 m` divided by the IAU 2012 astronomical unit `149597870700 m`: `asin(R☉/au) = 0.00465047 rad`. Sources: IAU 2015 Resolution B3, <https://www.iau.org/static/resolutions/IAU2015_English.pdf>, and IAU 2012 Resolution B2, <https://www.iau.org/static/resolutions/IAU2012_English.pdf>. The shader rounds to three significant figures; this changes the half-angle by less than `0.01%`.
