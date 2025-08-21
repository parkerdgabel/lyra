import * as vscode from 'vscode';

export class LyraHoverProvider implements vscode.HoverProvider {
    private functionDocs: Map<string, HoverInfo>;

    constructor() {
        this.functionDocs = this.createFunctionDocs();
    }

    public provideHover(
        document: vscode.TextDocument,
        position: vscode.Position,
        token: vscode.CancellationToken
    ): vscode.ProviderResult<vscode.Hover> {
        const wordRange = document.getWordRangeAtPosition(position);
        if (!wordRange) {
            return undefined;
        }

        const word = document.getText(wordRange);
        const hoverInfo = this.functionDocs.get(word);

        if (hoverInfo) {
            const markdownString = new vscode.MarkdownString();
            markdownString.isTrusted = true;
            markdownString.supportHtml = true;

            // Function signature
            markdownString.appendCodeblock(hoverInfo.signature, 'lyra');

            // Description
            markdownString.appendMarkdown(`\n${hoverInfo.description}\n\n`);

            // Parameters
            if (hoverInfo.parameters && hoverInfo.parameters.length > 0) {
                markdownString.appendMarkdown('**Parameters:**\n');
                hoverInfo.parameters.forEach(param => {
                    markdownString.appendMarkdown(`- \`${param.name}\`: ${param.description}\n`);
                });
                markdownString.appendMarkdown('\n');
            }

            // Examples
            if (hoverInfo.examples && hoverInfo.examples.length > 0) {
                markdownString.appendMarkdown('**Examples:**\n');
                hoverInfo.examples.forEach(example => {
                    markdownString.appendCodeblock(example.code, 'lyra');
                    if (example.result) {
                        markdownString.appendMarkdown(`Result: \`${example.result}\`\n\n`);
                    }
                });
            }

            // See also
            if (hoverInfo.seeAlso && hoverInfo.seeAlso.length > 0) {
                markdownString.appendMarkdown('**See also:** ');
                markdownString.appendMarkdown(hoverInfo.seeAlso.join(', '));
            }

            return new vscode.Hover(markdownString, wordRange);
        }

        return undefined;
    }

    public openFunctionDocs(functionName: string): void {
        const hoverInfo = this.functionDocs.get(functionName);
        if (hoverInfo) {
            // Create a new document with documentation
            const docContent = this.formatDocumentationAsMarkdown(functionName, hoverInfo);
            vscode.workspace.openTextDocument({
                content: docContent,
                language: 'markdown'
            }).then(doc => {
                vscode.window.showTextDocument(doc);
            });
        } else {
            vscode.window.showInformationMessage(`No documentation found for '${functionName}'`);
        }
    }

    private formatDocumentationAsMarkdown(functionName: string, info: HoverInfo): string {
        let content = `# ${functionName}\n\n`;
        
        content += `## Signature\n\`\`\`lyra\n${info.signature}\n\`\`\`\n\n`;
        
        content += `## Description\n${info.description}\n\n`;
        
        if (info.parameters && info.parameters.length > 0) {
            content += `## Parameters\n`;
            info.parameters.forEach(param => {
                content += `- **${param.name}**: ${param.description}\n`;
            });
            content += '\n';
        }
        
        if (info.examples && info.examples.length > 0) {
            content += `## Examples\n`;
            info.examples.forEach((example, index) => {
                content += `### Example ${index + 1}\n`;
                content += `\`\`\`lyra\n${example.code}\n\`\`\`\n`;
                if (example.result) {
                    content += `Result: \`${example.result}\`\n\n`;
                }
            });
        }
        
        if (info.seeAlso && info.seeAlso.length > 0) {
            content += `## See Also\n${info.seeAlso.join(', ')}\n`;
        }
        
        return content;
    }

    private createFunctionDocs(): Map<string, HoverInfo> {
        const docs = new Map<string, HoverInfo>();

        // Mathematical functions
        docs.set('Sin', {
            signature: 'Sin[x]',
            description: 'Computes the sine of x (x in radians).',
            parameters: [
                { name: 'x', description: 'Numeric expression in radians' }
            ],
            examples: [
                { code: 'Sin[Pi/2]', result: '1' },
                { code: 'Sin[0]', result: '0' },
                { code: 'Sin[Pi/6]', result: '1/2' }
            ],
            seeAlso: ['Cos', 'Tan', 'ArcSin']
        });

        docs.set('Cos', {
            signature: 'Cos[x]',
            description: 'Computes the cosine of x (x in radians).',
            parameters: [
                { name: 'x', description: 'Numeric expression in radians' }
            ],
            examples: [
                { code: 'Cos[0]', result: '1' },
                { code: 'Cos[Pi/2]', result: '0' },
                { code: 'Cos[Pi]', result: '-1' }
            ],
            seeAlso: ['Sin', 'Tan', 'ArcCos']
        });

        docs.set('Log', {
            signature: 'Log[x]',
            description: 'Computes the natural logarithm (base e) of x.',
            parameters: [
                { name: 'x', description: 'Positive real number' }
            ],
            examples: [
                { code: 'Log[E]', result: '1' },
                { code: 'Log[1]', result: '0' },
                { code: 'Log[E^2]', result: '2' }
            ],
            seeAlso: ['Exp', 'Log10']
        });

        docs.set('Exp', {
            signature: 'Exp[x]',
            description: 'Computes the exponential function e^x.',
            parameters: [
                { name: 'x', description: 'Numeric expression' }
            ],
            examples: [
                { code: 'Exp[0]', result: '1' },
                { code: 'Exp[1]', result: 'E' },
                { code: 'Exp[Log[x]]', result: 'x' }
            ],
            seeAlso: ['Log', 'Power']
        });

        // List functions
        docs.set('Length', {
            signature: 'Length[list]',
            description: 'Returns the number of elements in a list.',
            parameters: [
                { name: 'list', description: 'List expression' }
            ],
            examples: [
                { code: 'Length[{1, 2, 3}]', result: '3' },
                { code: 'Length[{}]', result: '0' },
                { code: 'Length[Range[10]]', result: '10' }
            ],
            seeAlso: ['First', 'Last', 'Take', 'Drop']
        });

        docs.set('Map', {
            signature: 'Map[f, list]',
            description: 'Applies function f to each element of list.',
            parameters: [
                { name: 'f', description: 'Function to apply' },
                { name: 'list', description: 'List to process' }
            ],
            examples: [
                { code: 'Map[Square, {1, 2, 3}]', result: '{1, 4, 9}' },
                { code: 'Map[Sin, {0, Pi/2, Pi}]', result: '{0, 1, 0}' }
            ],
            seeAlso: ['Apply', 'Select', 'Cases']
        });

        docs.set('Select', {
            signature: 'Select[list, predicate]',
            description: 'Returns elements of list for which predicate returns True.',
            parameters: [
                { name: 'list', description: 'List to filter' },
                { name: 'predicate', description: 'Boolean function' }
            ],
            examples: [
                { code: 'Select[{1, 2, 3, 4, 5}, EvenQ]', result: '{2, 4}' },
                { code: 'Select[{1, -2, 3, -4}, Positive]', result: '{1, 3}' }
            ],
            seeAlso: ['Cases', 'DeleteCases', 'Count']
        });

        // Control structures
        docs.set('If', {
            signature: 'If[condition, trueExpr, falseExpr]',
            description: 'Evaluates condition; if True, returns trueExpr, otherwise falseExpr.',
            parameters: [
                { name: 'condition', description: 'Boolean expression' },
                { name: 'trueExpr', description: 'Expression to evaluate if condition is True' },
                { name: 'falseExpr', description: 'Expression to evaluate if condition is False' }
            ],
            examples: [
                { code: 'If[x > 0, "positive", "non-positive"]' },
                { code: 'If[EvenQ[n], n/2, 3*n + 1]' }
            ],
            seeAlso: ['Which', 'Switch', 'Piecewise']
        });

        docs.set('Which', {
            signature: 'Which[cond1, expr1, cond2, expr2, ...]',
            description: 'Evaluates conditions in order and returns the expression corresponding to the first True condition.',
            parameters: [
                { name: 'cond1, cond2, ...', description: 'Boolean conditions' },
                { name: 'expr1, expr2, ...', description: 'Expressions to return' }
            ],
            examples: [
                { code: 'Which[x < 0, "negative", x == 0, "zero", True, "positive"]' }
            ],
            seeAlso: ['If', 'Switch', 'Piecewise']
        });

        // Pattern matching
        docs.set('MatchQ', {
            signature: 'MatchQ[expr, pattern]',
            description: 'Tests whether expr matches pattern.',
            parameters: [
                { name: 'expr', description: 'Expression to test' },
                { name: 'pattern', description: 'Pattern to match against' }
            ],
            examples: [
                { code: 'MatchQ[{1, 2, 3}, {___}]', result: 'True' },
                { code: 'MatchQ[f[x], f[_]]', result: 'True' }
            ],
            seeAlso: ['Cases', 'Replace', 'Count']
        });

        docs.set('Cases', {
            signature: 'Cases[expr, pattern]',
            description: 'Returns all subexpressions of expr that match pattern.',
            parameters: [
                { name: 'expr', description: 'Expression to search' },
                { name: 'pattern', description: 'Pattern to find' }
            ],
            examples: [
                { code: 'Cases[{1, 2, a, 3, b}, _Integer]', result: '{1, 2, 3}' },
                { code: 'Cases[{f[1], g[2], f[3]}, f[x_]]', result: '{f[1], f[3]}' }
            ],
            seeAlso: ['Select', 'Count', 'Position']
        });

        // Constants
        docs.set('Pi', {
            signature: 'Pi',
            description: 'The mathematical constant π ≈ 3.14159.',
            parameters: [],
            examples: [
                { code: 'N[Pi]', result: '3.14159' },
                { code: 'Sin[Pi/2]', result: '1' }
            ],
            seeAlso: ['E', 'GoldenRatio', 'EulerGamma']
        });

        docs.set('E', {
            signature: 'E',
            description: 'The mathematical constant e ≈ 2.71828 (base of natural logarithm).',
            parameters: [],
            examples: [
                { code: 'N[E]', result: '2.71828' },
                { code: 'Log[E]', result: '1' }
            ],
            seeAlso: ['Pi', 'Log', 'Exp']
        });

        return docs;
    }
}

interface HoverInfo {
    signature: string;
    description: string;
    parameters?: Array<{ name: string; description: string }>;
    examples?: Array<{ code: string; result?: string }>;
    seeAlso?: string[];
}