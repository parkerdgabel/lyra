import * as vscode from 'vscode';

export class LyraFormattingProvider implements vscode.DocumentFormattingEditProvider {
    
    public provideDocumentFormattingEdits(
        document: vscode.TextDocument,
        options: vscode.FormattingOptions,
        token: vscode.CancellationToken
    ): vscode.ProviderResult<vscode.TextEdit[]> {
        const edits: vscode.TextEdit[] = [];
        const indentSize = options.insertSpaces ? options.tabSize : 1;
        const indentChar = options.insertSpaces ? ' '.repeat(indentSize) : '\t';

        let indentLevel = 0;
        let inMultilineExpression = false;
        let bracketStack: string[] = [];

        for (let i = 0; i < document.lineCount; i++) {
            const line = document.lineAt(i);
            const originalText = line.text;
            const trimmedText = originalText.trim();

            if (trimmedText === '') {
                continue; // Skip empty lines
            }

            // Calculate current indent level based on brackets
            const currentIndentLevel = this.calculateIndentLevel(trimmedText, bracketStack);
            
            // Format the line
            const formattedText = this.formatLine(trimmedText, currentIndentLevel, indentChar);
            
            if (formattedText !== originalText) {
                const range = new vscode.Range(
                    new vscode.Position(i, 0),
                    new vscode.Position(i, originalText.length)
                );
                edits.push(new vscode.TextEdit(range, formattedText));
            }
        }

        return edits;
    }

    public async formatDocument(document: vscode.TextDocument): Promise<void> {
        const edits = this.provideDocumentFormattingEdits(
            document,
            {
                insertSpaces: true,
                tabSize: 2
            },
            new vscode.CancellationTokenSource().token
        );

        if (edits && edits.length > 0) {
            const workspaceEdit = new vscode.WorkspaceEdit();
            workspaceEdit.set(document.uri, edits);
            await vscode.workspace.applyEdit(workspaceEdit);
        }
    }

    private calculateIndentLevel(line: string, bracketStack: string[]): number {
        let indentLevel = bracketStack.length;
        
        // Count opening brackets at the beginning of line
        for (const char of line) {
            if (char === '[' || char === '{' || char === '(') {
                bracketStack.push(char);
            } else if (char === ']' || char === '}' || char === ')') {
                if (bracketStack.length > 0) {
                    const lastBracket = bracketStack.pop();
                    // Check for matching brackets
                    if (!this.isMatchingBracket(lastBracket!, char)) {
                        bracketStack.push(lastBracket!); // Put it back if not matching
                    }
                }
            } else if (char !== ' ' && char !== '\t') {
                break; // Stop at first non-whitespace, non-bracket character
            }
        }

        return Math.max(0, indentLevel);
    }

    private isMatchingBracket(open: string, close: string): boolean {
        const pairs: { [key: string]: string } = {
            '[': ']',
            '{': '}',
            '(': ')'
        };
        return pairs[open] === close;
    }

    private formatLine(line: string, indentLevel: number, indentChar: string): string {
        let formatted = indentChar.repeat(indentLevel) + line.trim();

        // Add spacing around operators
        formatted = this.addOperatorSpacing(formatted);

        // Format function calls
        formatted = this.formatFunctionCalls(formatted);

        // Format lists
        formatted = this.formatLists(formatted);

        return formatted;
    }

    private addOperatorSpacing(line: string): string {
        // Add spaces around assignment operators
        line = line.replace(/([^:])=([^=])/g, '$1 = $2');
        line = line.replace(/:=/g, ' := ');
        
        // Add spaces around comparison operators
        line = line.replace(/([^=!<>])==/g, '$1 == ');
        line = line.replace(/!=/g, ' != ');
        line = line.replace(/([^<])<=([^>])/g, '$1 <= $2');
        line = line.replace(/([^>])>=([^<])/g, '$1 >= $2');
        
        // Add spaces around rule operators
        line = line.replace(/([^-])->([^>])/g, '$1 -> $2');
        line = line.replace(/:>/g, ' :> ');
        line = line.replace(/\/\./g, ' /. ');
        
        // Add spaces around logical operators
        line = line.replace(/&&/g, ' && ');
        line = line.replace(/\|\|/g, ' || ');
        
        // Clean up multiple spaces
        line = line.replace(/\s+/g, ' ');
        
        return line;
    }

    private formatFunctionCalls(line: string): string {
        // Remove space before opening bracket in function calls
        line = line.replace(/([a-zA-Z_][a-zA-Z0-9_]*)\s+\[/g, '$1[');
        
        // Add space after commas in parameter lists
        line = line.replace(/,(?!\s)/g, ', ');
        
        return line;
    }

    private formatLists(line: string): string {
        // Add space after commas in lists
        line = line.replace(/,(?![}\s])/g, ', ');
        
        // Add space after opening brace and before closing brace
        line = line.replace(/\{(?!\s)/g, '{ ');
        line = line.replace(/(?<!\s)\}/g, ' }');
        
        // But not for empty lists
        line = line.replace(/\{\s+\}/g, '{}');
        
        return line;
    }
}