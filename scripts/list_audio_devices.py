"""Probe: list WASAPI loopback devices so we can confirm system-audio capture works.

Loopback devices let us record what comes OUT of your speakers (i.e. the other
people in a meeting), not just your microphone. If this prints a loopback device,
the hardest part of the project is already solved.
"""

import pyaudiowpatch as pyaudio


def main() -> None:
    with pyaudio.PyAudio() as p:
        # The default WASAPI output whose loopback we'd normally record.
        try:
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            print("WASAPI not available on this system.")
            return

        default_speakers = p.get_device_info_by_index(
            wasapi_info["defaultOutputDevice"]
        )
        print(f"Default output device: {default_speakers['name']}")
        print("-" * 60)

        print("Loopback devices (these capture system audio):")
        found = False
        for lb in p.get_loopback_device_info_generator():
            found = True
            print(f"  [{lb['index']:>3}] {lb['name']}")
            print(f"        channels={lb['maxInputChannels']} "
                  f"rate={int(lb['defaultSampleRate'])}Hz")

        if not found:
            print("  (none found)")

        print("-" * 60)
        try:
            default_mic = p.get_default_input_device_info()
            default_mic_index = default_mic["index"]
        except OSError:
            default_mic_index = None

        print("Microphones / input devices (pass the index as --mic-index):")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info["maxInputChannels"] < 1 or info.get("isLoopbackDevice"):
                continue
            marker = "  <- default" if i == default_mic_index else ""
            print(f"  [{i:>3}] {info['name']}{marker}")


if __name__ == "__main__":
    main()
