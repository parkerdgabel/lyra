import * as vscode from 'vscode';
import * as cp from 'child_process';
import * as path from 'path';

export class LyraReplProvider {
    private terminal: vscode.Terminal | undefined;
    private replProcess: cp.ChildProcess | undefined;
    private context: vscode.ExtensionContext;

    constructor(context: vscode.ExtensionContext) {
        this.context = context;
    }

    public async startRepl(): Promise<void> {
        if (this.terminal) {
            this.terminal.show();
            return;
        }

        const config = vscode.workspace.getConfiguration('lyra');
        const lyraPath = config.get<string>('repl.path', 'lyra');
        const lyraArgs = config.get<string[]>('repl.args', []);

        try {
            // Check if Lyra executable exists
            await this.checkLyraInstallation(lyraPath);

            // Create terminal with Lyra REPL
            this.terminal = vscode.window.createTerminal({
                name: 'Lyra REPL',
                shellPath: lyraPath,
                shellArgs: lyraArgs,
                iconPath: new vscode.ThemeIcon('terminal')
            });

            this.terminal.show();

            // Handle terminal close
            vscode.window.onDidCloseTerminal(terminal => {
                if (terminal === this.terminal) {
                    this.terminal = undefined;
                }
            });

            vscode.window.showInformationMessage('Lyra REPL started successfully!');
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to start Lyra REPL: ${error}`);
        }
    }

    public stopRepl(): void {
        if (this.terminal) {
            this.terminal.dispose();
            this.terminal = undefined;
            vscode.window.showInformationMessage('Lyra REPL stopped.');
        } else {
            vscode.window.showWarningMessage('No Lyra REPL is currently running.');
        }
    }

    public async restartRepl(): Promise<void> {
        this.stopRepl();
        await new Promise(resolve => setTimeout(resolve, 500)); // Wait a bit
        await this.startRepl();
    }

    public async evaluateSelection(): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor found.');
            return;
        }

        const selection = editor.selection;
        const text = editor.document.getText(selection.isEmpty ? undefined : selection);

        if (!text.trim()) {
            vscode.window.showWarningMessage('No text selected or file is empty.');
            return;
        }

        await this.ensureReplRunning();
        this.sendToRepl(text);
    }

    public async evaluateFile(): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor found.');
            return;
        }

        if (editor.document.languageId !== 'lyra') {
            vscode.window.showWarningMessage('Current file is not a Lyra file.');
            return;
        }

        // Save file first if it has unsaved changes
        if (editor.document.isDirty) {
            await editor.document.save();
        }

        const filePath = editor.document.fileName;
        await this.ensureReplRunning();
        
        // Use Lyra's file loading command
        this.sendToRepl(`%load "${filePath}"`);
    }

    private async ensureReplRunning(): Promise<void> {
        if (!this.terminal) {
            await this.startRepl();
        }
    }

    private sendToRepl(code: string): void {
        if (!this.terminal) {
            vscode.window.showErrorMessage('Lyra REPL is not running.');
            return;
        }

        // Show the terminal
        this.terminal.show();

        // Split code into lines and send each line
        const lines = code.split('\n');
        for (const line of lines) {
            if (line.trim()) {
                this.terminal.sendText(line, false);
            }
        }
        
        // Send final newline to execute
        this.terminal.sendText('', true);
    }

    private async checkLyraInstallation(lyraPath: string): Promise<void> {
        return new Promise((resolve, reject) => {
            cp.exec(`"${lyraPath}" --version`, (error, stdout, stderr) => {
                if (error) {
                    reject(new Error(`Lyra executable not found at "${lyraPath}". Please check your configuration.`));
                } else {
                    resolve();
                }
            });
        });
    }

    public async executeExpression(expression: string): Promise<string> {
        return new Promise((resolve, reject) => {
            const config = vscode.workspace.getConfiguration('lyra');
            const lyraPath = config.get<string>('repl.path', 'lyra');

            const process = cp.spawn(lyraPath, ['-c', expression]);
            let output = '';
            let errorOutput = '';

            process.stdout.on('data', (data) => {
                output += data.toString();
            });

            process.stderr.on('data', (data) => {
                errorOutput += data.toString();
            });

            process.on('close', (code) => {
                if (code === 0) {
                    resolve(output.trim());
                } else {
                    reject(new Error(errorOutput || `Process exited with code ${code}`));
                }
            });

            process.on('error', (error) => {
                reject(error);
            });
        });
    }

    public dispose(): void {
        this.stopRepl();
    }
}