- make the process faster maybe my using more threads, like use a percentage of the available threads for each task
- in a lot of images that get "sorted" in autonomous mode, it actually matches a number that is not in the sticker, this cant happen because then it means we cant trust the autonomous mode and if we cant trust it then we lose time and if we lose time there is no porpuse to it
- it seems to work great in images where the sticker and motorcycle are closer to the camera, but in images where the motorcycle is farther away and the sticker is smaller, it struggles more, so we should build a system to still read the number on the sticker.
- show how a sticker a normal image normaly are.
- see how to handle images where multple motorcycles are present
- add an option to also match the number of the event edition, for example "26" because a lot of people have multiple stickers from different editions

- use scc for concorrent containres and rapidhash for hashing, write performant code in general, if you have to write unsafe code or simd code, do so. Use rayon for parallelism.

the app seems to work well, look at the code base, at the plan, and improve the little things, and
improve performance, this app will be compiled on the machine that it will run, so it will be
compiled with "-Ctarget-cpu=native", so if you have to write unsafe code or simd code, do so,
performance is critical, also in the numbering part, going to next seems fast, but going prev seems
slow, there should atleast be one prev in cache. Also sometimes the number the OCR needs to find is
quite small and it seems the image that is fed to the OCR is has its quality reduced, but it makes
next to impossible to see the smaller numbers
