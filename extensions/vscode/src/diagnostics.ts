import * as vscode from 'vscode';

export class LyraDiagnosticsProvider {
    private diagnosticsCollection: vscode.DiagnosticCollection;
    private timeoutId: NodeJS.Timeout | undefined;

    constructor() {
        this.diagnosticsCollection = vscode.languages.createDiagnosticCollection('lyra');
    }

    public updateDiagnostics(document: vscode.TextDocument): void {
        // Debounce diagnostics updates
        if (this.timeoutId) {
            clearTimeout(this.timeoutId);
        }

        this.timeoutId = setTimeout(() => {
            this.performDiagnostics(document);
        }, 500);
    }

    private performDiagnostics(document: vscode.TextDocument): void {
        const diagnostics: vscode.Diagnostic[] = [];

        for (let i = 0; i < document.lineCount; i++) {
            const line = document.lineAt(i);
            const text = line.text;

            // Check for syntax errors
            diagnostics.push(...this.checkSyntaxErrors(line, i));
            
            // Check for bracket balance
            diagnostics.push(...this.checkBracketBalance(line, i));
            
            // Check for undefined functions
            diagnostics.push(...this.checkUndefinedFunctions(line, i));
            
            // Check for pattern errors
            diagnostics.push(...this.checkPatternErrors(line, i));
        }

        this.diagnosticsCollection.set(document.uri, diagnostics);
    }

    private checkSyntaxErrors(line: vscode.TextLine, lineNumber: number): vscode.Diagnostic[] {
        const diagnostics: vscode.Diagnostic[] = [];
        const text = line.text;

        // Check for invalid operators
        const invalidOperators = text.match(/[=!<>]{3,}|[&|]{3,}/g);
        if (invalidOperators) {
            invalidOperators.forEach(op => {
                const index = text.indexOf(op);
                const range = new vscode.Range(
                    lineNumber, index,
                    lineNumber, index + op.length
                );
                
                diagnostics.push(new vscode.Diagnostic(
                    range,
                    `Invalid operator: ${op}`,
                    vscode.DiagnosticSeverity.Error
                ));
            });
        }

        // Check for unterminated strings
        const stringMatches = text.match(/"[^"]*$/g) || text.match(/'[^']*$/g);
        if (stringMatches) {
            stringMatches.forEach(str => {
                const index = text.lastIndexOf(str);
                const range = new vscode.Range(
                    lineNumber, index,
                    lineNumber, text.length
                );
                
                diagnostics.push(new vscode.Diagnostic(
                    range,
                    'Unterminated string literal',
                    vscode.DiagnosticSeverity.Error
                ));
            });
        }

        // Check for invalid assignment patterns
        const invalidAssignments = text.match(/\d+\s*:?=/g);
        if (invalidAssignments) {
            invalidAssignments.forEach(assignment => {
                const index = text.indexOf(assignment);
                const range = new vscode.Range(
                    lineNumber, index,
                    lineNumber, index + assignment.length
                );
                
                diagnostics.push(new vscode.Diagnostic(
                    range,
                    'Cannot assign to numeric literal',
                    vscode.DiagnosticSeverity.Error
                ));
            });
        }

        return diagnostics;
    }

    private checkBracketBalance(line: vscode.TextLine, lineNumber: number): vscode.Diagnostic[] {
        const diagnostics: vscode.Diagnostic[] = [];
        const text = line.text;
        const brackets = { '[': ']', '{': '}', '(': ')' };
        const stack: Array<{ char: string; index: number }> = [];

        for (let i = 0; i < text.length; i++) {
            const char = text[i];
            
            if (Object.keys(brackets).includes(char)) {
                stack.push({ char, index: i });
            } else if (Object.values(brackets).includes(char)) {
                if (stack.length === 0) {
                    // Unmatched closing bracket
                    const range = new vscode.Range(lineNumber, i, lineNumber, i + 1);
                    diagnostics.push(new vscode.Diagnostic(
                        range,
                        `Unmatched closing bracket: ${char}`,
                        vscode.DiagnosticSeverity.Error
                    ));
                } else {
                    const last = stack.pop()!;
                    const expectedClosing = brackets[last.char as keyof typeof brackets];
                    
                    if (char !== expectedClosing) {
                        // Mismatched bracket
                        const range = new vscode.Range(lineNumber, i, lineNumber, i + 1);
                        diagnostics.push(new vscode.Diagnostic(
                            range,
                            `Expected '${expectedClosing}' but found '${char}'`,
                            vscode.DiagnosticSeverity.Error
                        ));
                    }
                }
            }
        }

        // Check for unclosed brackets
        stack.forEach(bracket => {
            const range = new vscode.Range(lineNumber, bracket.index, lineNumber, bracket.index + 1);
            const expectedClosing = brackets[bracket.char as keyof typeof brackets];
            diagnostics.push(new vscode.Diagnostic(
                range,
                `Unclosed bracket: expected '${expectedClosing}'`,
                vscode.DiagnosticSeverity.Error
            ));
        });

        return diagnostics;
    }

    private checkUndefinedFunctions(line: vscode.TextLine, lineNumber: number): vscode.Diagnostic[] {
        const diagnostics: vscode.Diagnostic[] = [];
        const text = line.text;

        // Known builtin functions
        const builtinFunctions = new Set([
            'Sin', 'Cos', 'Tan', 'ArcSin', 'ArcCos', 'ArcTan',
            'Log', 'Exp', 'Sqrt', 'Abs', 'Floor', 'Ceil', 'Round',
            'Max', 'Min', 'Sum', 'Product', 'Mean', 'Median',
            'Length', 'First', 'Last', 'Rest', 'Most', 'Take', 'Drop',
            'Map', 'Apply', 'Select', 'Cases', 'Count', 'Sort', 'Reverse',
            'If', 'Which', 'Switch', 'While', 'For', 'Do', 'Module', 'Block',
            'MatchQ', 'Replace', 'ReplaceAll', 'Print', 'Echo'
        ]);

        // Find function calls
        const functionCallRegex = /([A-Z][a-zA-Z0-9]*)\s*\[/g;
        let match;

        while ((match = functionCallRegex.exec(text)) !== null) {
            const functionName = match[1];
            
            if (!builtinFunctions.has(functionName)) {
                // Check if it's defined in the document (simple check)
                const isDefinedInDoc = this.isFunctionDefinedInDocument(functionName);
                
                if (!isDefinedInDoc) {
                    const range = new vscode.Range(
                        lineNumber, match.index,
                        lineNumber, match.index + functionName.length
                    );
                    
                    diagnostics.push(new vscode.Diagnostic(
                        range,
                        `Undefined function: ${functionName}`,
                        vscode.DiagnosticSeverity.Warning
                    ));
                }
            }
        }

        return diagnostics;
    }

    private checkPatternErrors(line: vscode.TextLine, lineNumber: number): vscode.Diagnostic[] {
        const diagnostics: vscode.Diagnostic[] = [];
        const text = line.text;

        // Check for invalid pattern syntax
        const invalidPatterns = text.match(/_{4,}|_[^a-zA-Z_]/g);
        if (invalidPatterns) {
            invalidPatterns.forEach(pattern => {
                const index = text.indexOf(pattern);
                const range = new vscode.Range(
                    lineNumber, index,
                    lineNumber, index + pattern.length
                );
                
                diagnostics.push(new vscode.Diagnostic(
                    range,
                    `Invalid pattern syntax: ${pattern}`,
                    vscode.DiagnosticSeverity.Error
                ));
            });
        }

        // Check for invalid rule syntax
        const invalidRules = text.match(/->-|:>>/g);
        if (invalidRules) {
            invalidRules.forEach(rule => {
                const index = text.indexOf(rule);
                const range = new vscode.Range(
                    lineNumber, index,
                    lineNumber, index + rule.length
                );
                
                diagnostics.push(new vscode.Diagnostic(
                    range,
                    `Invalid rule syntax: ${rule}`,
                    vscode.DiagnosticSeverity.Error
                ));
            });
        }

        return diagnostics;
    }

    private isFunctionDefinedInDocument(functionName: string): boolean {
        // This is a simplified check - in a real implementation,
        // you'd want to parse the entire document or use LSP
        const activeEditor = vscode.window.activeTextEditor;
        if (!activeEditor) return false;

        const text = activeEditor.document.getText();
        const definitionRegex = new RegExp(`${functionName}\\s*\\[[^\\]]*\\]\\s*:=`, 'g');
        
        return definitionRegex.test(text);
    }

    public dispose(): void {
        if (this.timeoutId) {
            clearTimeout(this.timeoutId);
        }
        this.diagnosticsCollection.dispose();
    }
}