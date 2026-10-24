import io
import json

import pytest

from lure_tts.cli import CliError, build_parser, main, parse_opts, parse_say
from lure_tts.defaults import Sampling


def say(*argv, stdin=None):
    return parse_say(build_parser().parse_args(["say", *argv]), stdin=stdin)


def test_say_voice_defaults():
    r = say("Hello there.", "--voice", "willow", "-o", "x.wav")
    assert r.text == "Hello there." and r.voice == "willow" and r.out == "x.wav"
    assert r.sampling == Sampling() and r.sampling.repetition_penalty == 1.05
    assert r.instruct is None and r.backend == "auto"


def test_say_sampling_and_direction_alias():
    r = say("Hi.", "--voice", "v", "-o", "-", "--direction", " Warm. ", "--temperature", "0.7", "--top-k", "30",
            "--repetition-penalty", "1.1", "--seed", "7", "--max-new-tokens", "100")
    assert r.instruct == "Warm."
    assert r.sampling.temperature == 0.7 and r.sampling.top_k == 30 and r.sampling.repetition_penalty == 1.1
    assert r.sampling.top_p == 1.0 and r.seed == 7 and r.sampling.max_new_tokens == 100


def test_say_stdin_text():
    r = say("-", "--voice", "v", "--play", stdin=io.StringIO("(laugh) from stdin."))
    assert r.text == "(laugh) from stdin." and r.play


def test_say_ref_voice_and_save():
    r = say("Hi.", "--ref-wav", "a.wav", "--ref-text", "ref", "--save-voice", "me", "-o", "o.wav")
    assert r.ref_wav == "a.wav" and r.save_voice == "me" and r.voice is None


@pytest.mark.parametrize("argv", [
    ["Hi.", "-o", "o.wav"],                                            # no voice
    ["Hi.", "--voice", "v"],                                           # no output
    ["Hi.", "--voice", "v", "--ref-wav", "a.wav", "-o", "o.wav"],      # both voice kinds
    ["Hi.", "--ref-wav", "a.wav", "-o", "o.wav"],                      # missing ref text
    ["   ", "--voice", "v", "-o", "o.wav"],                            # empty text
    ["Hi.", "--voice", "v", "-o", "o", "--save-voice", "x"],           # save without ref
])
def test_say_errors(argv):
    with pytest.raises(CliError):
        say(*argv)


def test_blank_direction_is_none():
    assert say("Hi.", "--voice", "v", "-o", "o", "--instruct", "   ").instruct is None


def test_opts():
    assert parse_opts(["compile_cp=1", "pipeline=off", "cache_mb=512", "x=abc", "f=0.5"]) == {
        "compile_cp": True, "pipeline": False, "cache_mb": 512, "x": "abc", "f": 0.5}
    with pytest.raises(CliError):
        parse_opts(["novalue"])


def test_prompt_command_json(tokdir, capsys):
    assert main(["prompt", "(sigh) Fine. I'm in.", "--direction", "Tired.", "--model", tokdir, "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["instruct_ids"][0] == 151644 and out["spans"]["total"] == len(out["layout"])


def test_say_end_to_end_fake(tokdir, tmp_path):
    import soundfile as sf
    import numpy as np
    ref = tmp_path / "ref.wav"
    sf.write(ref, np.zeros(24000, np.float32), 24000)
    out = tmp_path / "o.wav"
    rc = main(["say", "Hello there, this is a test. (laugh) Second sentence here.", "--ref-wav", str(ref), "--ref-text", "Ref.",
               "--save-voice", "me", "-o", str(out), "--backend", "fake", "--model", tokdir, "--cache", str(tmp_path)])
    assert rc == 0
    x, sr = sf.read(out)
    assert sr == 24000 and len(x) > 0
    rc = main(["say", "Again.", "--voice", "me", "-o", str(out), "--backend", "fake", "--model", tokdir, "--cache", str(tmp_path)])
    assert rc == 0
