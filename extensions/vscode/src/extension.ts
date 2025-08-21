import * as vscode from 'vscode';
import { LyraReplProvider } from './repl';
import { LyraExportProvider } from './export';
import { LyraCompletionProvider } from './completion';
import { LyraHoverProvider } from './hover';
import { LyraFormattingProvider } from './formatting';
import { LyraDiagnosticsProvider } from './diagnostics';

export function activate(context: vscode.ExtensionContext) {
    console.log('Lyra Language Extension is now active!');

    // Initialize providers
    const replProvider = new LyraReplProvider(context);
    const exportProvider = new LyraExportProvider(context);
    const completionProvider = new LyraCompletionProvider();
    const hoverProvider = new LyraHoverProvider();
    const formattingProvider = new LyraFormattingProvider();
    const diagnosticsProvider = new LyraDiagnosticsProvider();

    // Register REPL commands
    context.subscriptions.push(
        vscode.commands.registerCommand('lyra.repl.start', () => replProvider.startRepl()),
        vscode.commands.registerCommand('lyra.repl.stop', () => replProvider.stopRepl()),
        vscode.commands.registerCommand('lyra.repl.restart', () => replProvider.restartRepl()),
        vscode.commands.registerCommand('lyra.repl.evaluateSelection', () => replProvider.evaluateSelection()),
        vscode.commands.registerCommand('lyra.repl.evaluateFile', () => replProvider.evaluateFile())
    );

    // Register export commands
    context.subscriptions.push(
        vscode.commands.registerCommand('lyra.export.jupyter', () => exportProvider.exportToJupyter()),
        vscode.commands.registerCommand('lyra.export.latex', () => exportProvider.exportToLatex()),
        vscode.commands.registerCommand('lyra.export.html', () => exportProvider.exportToHtml())
    );

    // Register documentation command
    context.subscriptions.push(
        vscode.commands.registerCommand('lyra.docs.openFunction', () => {
            const editor = vscode.window.activeTextEditor;
            if (editor) {
                const position = editor.selection.active;
                const wordRange = editor.document.getWordRangeAtPosition(position);
                if (wordRange) {
                    const word = editor.document.getText(wordRange);
                    hoverProvider.openFunctionDocs(word);
                }
            }
        })
    );

    // Register formatting command
    context.subscriptions.push(
        vscode.commands.registerCommand('lyra.format.document', () => {
            const editor = vscode.window.activeTextEditor;
            if (editor && editor.document.languageId === 'lyra') {
                formattingProvider.formatDocument(editor.document);
            }
        })
    );

    // Register language features
    const lyraSelector = { scheme: 'file', language: 'lyra' };

    // Completion provider
    context.subscriptions.push(
        vscode.languages.registerCompletionItemProvider(
            lyraSelector,
            completionProvider,
            '[', '(', ',', ' '
        )
    );

    // Hover provider
    context.subscriptions.push(
        vscode.languages.registerHoverProvider(lyraSelector, hoverProvider)
    );

    // Document formatting provider
    context.subscriptions.push(
        vscode.languages.registerDocumentFormattingEditProvider(lyraSelector, formattingProvider)
    );

    // Diagnostics provider
    context.subscriptions.push(
        vscode.workspace.onDidChangeTextDocument(event => {
            if (event.document.languageId === 'lyra') {
                diagnosticsProvider.updateDiagnostics(event.document);
            }
        })
    );

    // Auto-start REPL if enabled
    const config = vscode.workspace.getConfiguration('lyra');
    if (config.get('repl.autoStart')) {
        vscode.workspace.onDidOpenTextDocument(document => {
            if (document.languageId === 'lyra') {
                replProvider.startRepl();
            }
        });
    }

    // Status bar items
    const replStatusItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
    replStatusItem.text = "$(terminal) Lyra REPL";
    replStatusItem.command = 'lyra.repl.start';
    replStatusItem.tooltip = 'Start Lyra REPL';
    context.subscriptions.push(replStatusItem);

    // Show status bar for Lyra files
    vscode.window.onDidChangeActiveTextEditor(editor => {
        if (editor && editor.document.languageId === 'lyra') {
            replStatusItem.show();
        } else {
            replStatusItem.hide();
        }
    });

    // Welcome message
    vscode.window.showInformationMessage(
        'Lyra Language Extension activated! Use Ctrl+Shift+L to start REPL.',
        'Show Commands'
    ).then(selection => {
        if (selection === 'Show Commands') {
            vscode.commands.executeCommand('workbench.action.showCommands', 'Lyra:');
        }
    });
}

export function deactivate() {
    console.log('Lyra Language Extension is now deactivated.');
}