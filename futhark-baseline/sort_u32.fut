import "lib/github.com/diku-dk/sorts/radix_sort"

entry main (xs: []u32) : []u32 =
  radix_sort u32.num_bits u32.get_bit xs
