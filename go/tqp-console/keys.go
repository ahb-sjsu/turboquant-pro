// TurboQuant Pro console: terminal client.
// Copyright (c) 2026 Andrew H. Bond. MIT License.

package main

// Raw terminal input to key names: the names the engine's key functions take
// ("space", "up", "enter", "escape", "tab", or the character itself), plus
// "ctrl+c" and "ctrl+z", which in raw mode arrive as bytes, not signals.

// ParseKeys turns one burst of input into key names. A burst that ends inside an
// escape sequence is reported as incomplete, so the reader can wait briefly for
// the rest before deciding that a lone ESC was the Escape key.
func ParseKeys(b []byte) (keys []string, incomplete bool) {
	for i := 0; i < len(b); i++ {
		c := b[i]
		switch {
		case c == 0x1b:
			if i+1 >= len(b) {
				return keys, true // ESC alone, or the start of a sequence
			}
			n := b[i+1]
			if n != '[' && n != 'O' {
				keys = append(keys, "escape")
				continue
			}
			// CSI / SS3: parameters then one final byte in 0x40..0x7e
			j := i + 2
			for j < len(b) && (b[j] < 0x40 || b[j] > 0x7e) {
				j++
			}
			if j >= len(b) {
				return keys, true
			}
			switch b[j] {
			case 'A':
				keys = append(keys, "up")
			case 'B':
				keys = append(keys, "down")
			case 'C':
				keys = append(keys, "right")
			case 'D':
				keys = append(keys, "left")
			case 'Z':
				keys = append(keys, "backtab")
			}
			i = j
		case c == 0x03:
			keys = append(keys, "ctrl+c")
		case c == 0x1a:
			keys = append(keys, "ctrl+z")
		case c == '\r' || c == '\n':
			keys = append(keys, "enter")
		case c == '\t':
			keys = append(keys, "tab")
		case c == ' ':
			keys = append(keys, "space")
		case c >= 0x21 && c < 0x7f:
			keys = append(keys, string(rune(c)))
		}
	}
	return keys, false
}
