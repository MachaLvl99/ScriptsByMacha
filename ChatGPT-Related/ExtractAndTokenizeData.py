
import json
import nltk
nltk.download('punkt')

def TokenizeStringSimple(s):
    return nltk.word_tokenize(s)

def ExtractAndTokenizeConversations(InputFile, OutputFile):
    with open(InputFile, 'r') as f:
        data = json.load(f)

    ExtractedMessages = []
    for conversation in data:
        for NodeId, NodeData in conversation["mapping"].items():
            if NodeData is not None and "message" in NodeData and NodeData["message"] is not None and NodeData["message"]["content"] is not None and NodeData["message"]["content"]["content_type"] == "text":
                try:
                    TokenizedContent = TokenizeStringSimple(NodeData["message"]["content"]["parts"][0])
                    ExtractedMessages.append({
                        "id": NodeId,
                        "role": NodeData["message"]["author"]["role"],
                        "content": NodeData["message"]["content"]["parts"][0],
                        "tokens": TokenizedContent,
                        "parent": NodeData["parent"],
                        "children": NodeData["children"],
                    })
                except IndexError:
                    pass

    with open(OutputFile, 'w') as f:
        json.dump(ExtractedMessages, f, indent=4)

# Usage:
# ExtractAndTokenizeConversations("/path/to/input.file", "/path/to/output.file")
