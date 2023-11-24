bad_segmentations_tail = ["31011ade7c0a", "a596ad0b82aa", "13b4d6a01d27", "10469f6765bf",
                              "05e1944c3818", "a9a2f7fac455"] # non completed segmentations
bad_segmentations_tail = bad_segmentations_tail + [
    "4feda0596965", "df33ae359fb5"
] # partial, with some non-completed segmentations at the end, followed by really taking off watch. still excluded since the contents of taking off watch are not much

noisy_bad_segmentations = [
    "13b4d6a01d27", "4feda0596965", "60d31b0bec3b", "e4500e7e19e1", "f56824b503a0", "cf13ed7e457a"
] # see bad_series_notes.txt

#cf13ed7e457a, many repeating segments with unlabelled events. can be removed with matrix profile