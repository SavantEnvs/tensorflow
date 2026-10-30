// Turn LeakSanitizer off at build time; ASan's memory-corruption checks stay on.
extern "C" int __lsan_is_turned_off() { return 1; }
