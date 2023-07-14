
import json
import nltk
nltk.download('punkt')

def TokenizeStringSimple(s):
    return nltk.word_tokenize(s)

def FormatAndChunkMessages(InputFile, OutputFile):
    with open(InputFile, 'r') as f:
        data = json.load(f)

    BeginMessage = "[BEGIN_MESSAGE|{}]"
    EndMessage = "[END_MESSAGE]"
    BeginChunk = "[BEGIN_CHUNK|CONVERSATION_1|PART_{}]"
    EndChunk = "[END_CHUNK|CONVERSATION_1|PART_{}]"
    MaxTokensPerChunk = 4096

    Chunks = []
    CurrentChunk = []
    CurrentTokens = 0

    for message in data:
        FormattedMessage = f"{BeginMessage.format(message['role'].upper())} {message['content']} {EndMessage}"
        TokenizedMessage = TokenizeStringSimple(FormattedMessage)

        if CurrentTokens + len(TokenizedMessage) > MaxTokensPerChunk:
            CurrentChunk.append(EndChunk.format(len(Chunks) + 1))
            Chunks.append(' '.join(CurrentChunk))
            CurrentChunk = [BeginChunk.format(len(Chunks) + 2)]
            CurrentTokens = len(CurrentChunk[0])

        CurrentChunk.append(FormattedMessage)
        CurrentTokens += len(TokenizedMessage)

    if CurrentChunk:
        CurrentChunk.append(EndChunk.format(len(Chunks) + 1))
        Chunks.append(' '.join(CurrentChunk))

    with open(OutputFile, 'w') as f:
        json.dump(Chunks, f, indent=4)

# Usage:
# FormatAndChunkMessages("/path/to/input.file", "/path/to/output.file")
