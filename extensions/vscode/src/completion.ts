import * as vscode from 'vscode';

export class LyraCompletionProvider implements vscode.CompletionItemProvider {
    private builtinFunctions: Map<string, vscode.CompletionItem>;
    private mathConstants: Map<string, vscode.CompletionItem>;
    private controlStructures: Map<string, vscode.CompletionItem>;

    constructor() {
        this.builtinFunctions = this.createBuiltinFunctions();
        this.mathConstants = this.createMathConstants();
        this.controlStructures = this.createControlStructures();
    }

    public provideCompletionItems(
        document: vscode.TextDocument,
        position: vscode.Position,
        token: vscode.CancellationToken,
        context: vscode.CompletionContext
    ): vscode.ProviderResult<vscode.CompletionItem[]> {
        const line = document.lineAt(position).text;
        const linePrefix = line.substring(0, position.character);

        const completions: vscode.CompletionItem[] = [];

        // Add all builtin functions
        completions.push(...Array.from(this.builtinFunctions.values()));

        // Add math constants
        completions.push(...Array.from(this.mathConstants.values()));

        // Add control structures
        completions.push(...Array.from(this.controlStructures.values()));

        // Add context-specific completions
        if (linePrefix.endsWith('[')) {
            // Inside function brackets - suggest parameter patterns
            completions.push(...this.createParameterCompletions());
        }

        if (linePrefix.includes('/.')) {
            // Rule replacement context
            completions.push(...this.createRuleCompletions());
        }

        // Variable completions from document
        completions.push(...this.getVariableCompletions(document, position));

        return completions;
    }

    private createBuiltinFunctions(): Map<string, vscode.CompletionItem> {
        const functions = new Map<string, vscode.CompletionItem>();

        // Mathematical functions
        const mathFunctions = [
            { name: 'Sin', detail: 'Sin[x] - Sine function', snippet: 'Sin[${1:x}]' },
            { name: 'Cos', detail: 'Cos[x] - Cosine function', snippet: 'Cos[${1:x}]' },
            { name: 'Tan', detail: 'Tan[x] - Tangent function', snippet: 'Tan[${1:x}]' },
            { name: 'ArcSin', detail: 'ArcSin[x] - Inverse sine', snippet: 'ArcSin[${1:x}]' },
            { name: 'ArcCos', detail: 'ArcCos[x] - Inverse cosine', snippet: 'ArcCos[${1:x}]' },
            { name: 'ArcTan', detail: 'ArcTan[x] - Inverse tangent', snippet: 'ArcTan[${1:x}]' },
            { name: 'Log', detail: 'Log[x] - Natural logarithm', snippet: 'Log[${1:x}]' },
            { name: 'Exp', detail: 'Exp[x] - Exponential function', snippet: 'Exp[${1:x}]' },
            { name: 'Sqrt', detail: 'Sqrt[x] - Square root', snippet: 'Sqrt[${1:x}]' },
            { name: 'Abs', detail: 'Abs[x] - Absolute value', snippet: 'Abs[${1:x}]' },
            { name: 'Floor', detail: 'Floor[x] - Floor function', snippet: 'Floor[${1:x}]' },
            { name: 'Ceil', detail: 'Ceil[x] - Ceiling function', snippet: 'Ceil[${1:x}]' },
            { name: 'Round', detail: 'Round[x] - Round to nearest integer', snippet: 'Round[${1:x}]' },
            { name: 'Max', detail: 'Max[x, y, ...] - Maximum value', snippet: 'Max[${1:x}, ${2:y}]' },
            { name: 'Min', detail: 'Min[x, y, ...] - Minimum value', snippet: 'Min[${1:x}, ${2:y}]' }
        ];

        // List functions
        const listFunctions = [
            { name: 'Length', detail: 'Length[list] - Length of list', snippet: 'Length[${1:list}]' },
            { name: 'First', detail: 'First[list] - First element', snippet: 'First[${1:list}]' },
            { name: 'Last', detail: 'Last[list] - Last element', snippet: 'Last[${1:list}]' },
            { name: 'Rest', detail: 'Rest[list] - All but first element', snippet: 'Rest[${1:list}]' },
            { name: 'Most', detail: 'Most[list] - All but last element', snippet: 'Most[${1:list}]' },
            { name: 'Take', detail: 'Take[list, n] - First n elements', snippet: 'Take[${1:list}, ${2:n}]' },
            { name: 'Drop', detail: 'Drop[list, n] - Drop first n elements', snippet: 'Drop[${1:list}, ${2:n}]' },
            { name: 'Join', detail: 'Join[list1, list2] - Concatenate lists', snippet: 'Join[${1:list1}, ${2:list2}]' },
            { name: 'Sort', detail: 'Sort[list] - Sort list', snippet: 'Sort[${1:list}]' },
            { name: 'Reverse', detail: 'Reverse[list] - Reverse list', snippet: 'Reverse[${1:list}]' },
            { name: 'Map', detail: 'Map[f, list] - Apply function to each element', snippet: 'Map[${1:function}, ${2:list}]' },
            { name: 'Select', detail: 'Select[list, predicate] - Filter list', snippet: 'Select[${1:list}, ${2:predicate}]' },
            { name: 'Apply', detail: 'Apply[f, args] - Apply function to arguments', snippet: 'Apply[${1:function}, ${2:args}]' },
            { name: 'Fold', detail: 'Fold[f, init, list] - Fold operation', snippet: 'Fold[${1:function}, ${2:init}, ${3:list}]' }
        ];

        // Pattern matching functions
        const patternFunctions = [
            { name: 'MatchQ', detail: 'MatchQ[expr, pattern] - Test if expression matches pattern', snippet: 'MatchQ[${1:expr}, ${2:pattern}]' },
            { name: 'Cases', detail: 'Cases[expr, pattern] - Extract matching cases', snippet: 'Cases[${1:expr}, ${2:pattern}]' },
            { name: 'Count', detail: 'Count[expr, pattern] - Count matches', snippet: 'Count[${1:expr}, ${2:pattern}]' },
            { name: 'Replace', detail: 'Replace[expr, rule] - Replace once', snippet: 'Replace[${1:expr}, ${2:rule}]' },
            { name: 'ReplaceAll', detail: 'ReplaceAll[expr, rules] - Replace all occurrences', snippet: 'ReplaceAll[${1:expr}, ${2:rules}]' }
        ];

        // Combine all functions
        const allFunctions = [...mathFunctions, ...listFunctions, ...patternFunctions];

        allFunctions.forEach(func => {
            const item = new vscode.CompletionItem(func.name, vscode.CompletionItemKind.Function);
            item.detail = func.detail;
            item.insertText = new vscode.SnippetString(func.snippet);
            item.documentation = new vscode.MarkdownString(func.detail);
            functions.set(func.name, item);
        });

        return functions;
    }

    private createMathConstants(): Map<string, vscode.CompletionItem> {
        const constants = new Map<string, vscode.CompletionItem>();

        const mathConstants = [
            { name: 'Pi', detail: 'π ≈ 3.14159', value: 'Pi' },
            { name: 'E', detail: 'Euler\'s number ≈ 2.71828', value: 'E' },
            { name: 'I', detail: 'Imaginary unit √(-1)', value: 'I' },
            { name: 'Infinity', detail: '∞', value: 'Infinity' },
            { name: 'GoldenRatio', detail: 'φ ≈ 1.618', value: 'GoldenRatio' },
            { name: 'EulerGamma', detail: 'γ ≈ 0.5772', value: 'EulerGamma' }
        ];

        // Greek letters
        const greekLetters = [
            'Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta',
            'Iota', 'Kappa', 'Lambda', 'Mu', 'Nu', 'Xi', 'Omicron', 'Pi', 'Rho',
            'Sigma', 'Tau', 'Upsilon', 'Phi', 'Chi', 'Psi', 'Omega'
        ];

        mathConstants.forEach(constant => {
            const item = new vscode.CompletionItem(constant.name, vscode.CompletionItemKind.Constant);
            item.detail = constant.detail;
            item.insertText = constant.value;
            item.documentation = new vscode.MarkdownString(`Mathematical constant: ${constant.detail}`);
            constants.set(constant.name, item);
        });

        greekLetters.forEach(letter => {
            const item = new vscode.CompletionItem(letter, vscode.CompletionItemKind.Constant);
            item.detail = `Greek letter ${letter}`;
            item.insertText = letter;
            item.documentation = new vscode.MarkdownString(`Greek letter: ${letter}`);
            constants.set(letter, item);
        });

        return constants;
    }

    private createControlStructures(): Map<string, vscode.CompletionItem> {
        const structures = new Map<string, vscode.CompletionItem>();

        const controlStructures = [
            {
                name: 'If',
                detail: 'If[condition, trueExpr, falseExpr]',
                snippet: 'If[${1:condition}, ${2:trueExpr}, ${3:falseExpr}]'
            },
            {
                name: 'Which',
                detail: 'Which[cond1, expr1, cond2, expr2, ...]',
                snippet: 'Which[\\n  ${1:condition1}, ${2:expr1},\\n  ${3:condition2}, ${4:expr2},\\n  True, ${5:defaultExpr}\\n]'
            },
            {
                name: 'For',
                detail: 'For[init, condition, increment, body]',
                snippet: 'For[${1:i} = ${2:start}, ${1:i} <= ${3:end}, ${1:i}++,\\n  ${4:body}\\n]'
            },
            {
                name: 'While',
                detail: 'While[condition, body]',
                snippet: 'While[${1:condition},\\n  ${2:body}\\n]'
            },
            {
                name: 'Module',
                detail: 'Module[{vars}, body]',
                snippet: 'Module[{${1:vars}},\\n  ${2:body}\\n]'
            },
            {
                name: 'Block',
                detail: 'Block[{vars}, body]',
                snippet: 'Block[{${1:vars}},\\n  ${2:body}\\n]'
            },
            {
                name: 'Function',
                detail: 'Function[{args}, body]',
                snippet: 'Function[{${1:args}}, ${2:body}]'
            }
        ];

        controlStructures.forEach(structure => {
            const item = new vscode.CompletionItem(structure.name, vscode.CompletionItemKind.Keyword);
            item.detail = structure.detail;
            item.insertText = new vscode.SnippetString(structure.snippet);
            item.documentation = new vscode.MarkdownString(`Control structure: ${structure.detail}`);
            structures.set(structure.name, item);
        });

        return structures;
    }

    private createParameterCompletions(): vscode.CompletionItem[] {
        const completions: vscode.CompletionItem[] = [];

        // Pattern completions
        const patterns = [
            { name: 'x_', detail: 'Named blank pattern', snippet: '${1:x}_' },
            { name: 'x_Integer', detail: 'Integer pattern', snippet: '${1:x}_Integer' },
            { name: 'x_Real', detail: 'Real number pattern', snippet: '${1:x}_Real' },
            { name: 'x_String', detail: 'String pattern', snippet: '${1:x}_String' },
            { name: 'x_List', detail: 'List pattern', snippet: '${1:x}_List' },
            { name: 'x__', detail: 'Sequence pattern', snippet: '${1:x}__' },
            { name: 'x___', detail: 'Null sequence pattern', snippet: '${1:x}___' }
        ];

        patterns.forEach(pattern => {
            const item = new vscode.CompletionItem(pattern.name, vscode.CompletionItemKind.TypeParameter);
            item.detail = pattern.detail;
            item.insertText = new vscode.SnippetString(pattern.snippet);
            item.documentation = new vscode.MarkdownString(`Pattern: ${pattern.detail}`);
            completions.push(item);
        });

        return completions;
    }

    private createRuleCompletions(): vscode.CompletionItem[] {
        const completions: vscode.CompletionItem[] = [];

        // Rule operators
        const rules = [
            { name: '->', detail: 'Replacement rule', snippet: ' -> ${1:replacement}' },
            { name: ':>', detail: 'Delayed rule', snippet: ' :> ${1:replacement}' }
        ];

        rules.forEach(rule => {
            const item = new vscode.CompletionItem(rule.name, vscode.CompletionItemKind.Operator);
            item.detail = rule.detail;
            item.insertText = new vscode.SnippetString(rule.snippet);
            item.documentation = new vscode.MarkdownString(`Rule operator: ${rule.detail}`);
            completions.push(item);
        });

        return completions;
    }

    private getVariableCompletions(document: vscode.TextDocument, position: vscode.Position): vscode.CompletionItem[] {
        const completions: vscode.CompletionItem[] = [];
        const variables = new Set<string>();

        // Scan document for variable assignments
        for (let i = 0; i < document.lineCount; i++) {
            const line = document.lineAt(i).text;
            
            // Match variable assignments: varName = value or varName := value
            const assignmentRegex = /([a-zA-Z][a-zA-Z0-9]*)\s*:?=/g;
            let match;
            
            while ((match = assignmentRegex.exec(line)) !== null) {
                const varName = match[1];
                if (!this.builtinFunctions.has(varName) && !this.mathConstants.has(varName)) {
                    variables.add(varName);
                }
            }

            // Match function definitions: funcName[args_] := body
            const functionRegex = /([a-zA-Z][a-zA-Z0-9]*)\[[^\]]*\]\s*:=/g;
            while ((match = functionRegex.exec(line)) !== null) {
                const funcName = match[1];
                variables.add(funcName);
            }
        }

        // Convert to completion items
        variables.forEach(varName => {
            const item = new vscode.CompletionItem(varName, vscode.CompletionItemKind.Variable);
            item.detail = 'User-defined variable';
            item.documentation = new vscode.MarkdownString(`Variable defined in this document`);
            completions.push(item);
        });

        return completions;
    }
}