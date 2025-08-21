import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';

export class LyraExportProvider {
    private context: vscode.ExtensionContext;

    constructor(context: vscode.ExtensionContext) {
        this.context = context;
    }

    public async exportToJupyter(): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor || editor.document.languageId !== 'lyra') {
            vscode.window.showWarningMessage('Please open a Lyra file to export.');
            return;
        }

        const outputPath = await this.getExportPath('jupyter', '.ipynb');
        if (!outputPath) return;

        try {
            await this.executeExport('jupyter', editor.document.fileName, outputPath);
            vscode.window.showInformationMessage(
                `Exported to Jupyter notebook: ${path.basename(outputPath)}`,
                'Open File'
            ).then(selection => {
                if (selection === 'Open File') {
                    vscode.env.openExternal(vscode.Uri.file(outputPath));
                }
            });
        } catch (error) {
            vscode.window.showErrorMessage(`Export failed: ${error}`);
        }
    }

    public async exportToLatex(): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor || editor.document.languageId !== 'lyra') {
            vscode.window.showWarningMessage('Please open a Lyra file to export.');
            return;
        }

        const outputPath = await this.getExportPath('latex', '.tex');
        if (!outputPath) return;

        try {
            await this.executeExport('latex', editor.document.fileName, outputPath);
            vscode.window.showInformationMessage(
                `Exported to LaTeX: ${path.basename(outputPath)}`,
                'Open File', 'Compile PDF'
            ).then(selection => {
                if (selection === 'Open File') {
                    vscode.workspace.openTextDocument(outputPath).then(doc => {
                        vscode.window.showTextDocument(doc);
                    });
                } else if (selection === 'Compile PDF') {
                    this.compilePDF(outputPath);
                }
            });
        } catch (error) {
            vscode.window.showErrorMessage(`Export failed: ${error}`);
        }
    }

    public async exportToHtml(): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor || editor.document.languageId !== 'lyra') {
            vscode.window.showWarningMessage('Please open a Lyra file to export.');
            return;
        }

        const outputPath = await this.getExportPath('html', '.html');
        if (!outputPath) return;

        try {
            await this.executeExport('html', editor.document.fileName, outputPath);
            vscode.window.showInformationMessage(
                `Exported to HTML: ${path.basename(outputPath)}`,
                'Open in Browser', 'Open File'
            ).then(selection => {
                if (selection === 'Open in Browser') {
                    vscode.env.openExternal(vscode.Uri.file(outputPath));
                } else if (selection === 'Open File') {
                    vscode.workspace.openTextDocument(outputPath).then(doc => {
                        vscode.window.showTextDocument(doc);
                    });
                }
            });
        } catch (error) {
            vscode.window.showErrorMessage(`Export failed: ${error}`);
        }
    }

    private async getExportPath(format: string, extension: string): Promise<string | undefined> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return undefined;

        const currentFile = editor.document.fileName;
        const baseName = path.basename(currentFile, path.extname(currentFile));
        const defaultPath = path.join(path.dirname(currentFile), `${baseName}${extension}`);

        const uri = await vscode.window.showSaveDialog({
            defaultUri: vscode.Uri.file(defaultPath),
            filters: this.getFileFilters(format)
        });

        return uri?.fsPath;
    }

    private getFileFilters(format: string): { [name: string]: string[] } {
        switch (format) {
            case 'jupyter':
                return {
                    'Jupyter Notebooks': ['ipynb'],
                    'All Files': ['*']
                };
            case 'latex':
                return {
                    'LaTeX Files': ['tex'],
                    'All Files': ['*']
                };
            case 'html':
                return {
                    'HTML Files': ['html', 'htm'],
                    'All Files': ['*']
                };
            default:
                return { 'All Files': ['*'] };
        }
    }

    private async executeExport(format: string, inputPath: string, outputPath: string): Promise<void> {
        return new Promise((resolve, reject) => {
            const config = vscode.workspace.getConfiguration('lyra');
            const lyraPath = config.get<string>('repl.path', 'lyra');

            // Use Lyra's export command
            const args = ['--export', format, '--input', inputPath, '--output', outputPath];
            
            const { spawn } = require('child_process');
            const process = spawn(lyraPath, args);

            let stderr = '';

            process.stderr.on('data', (data: Buffer) => {
                stderr += data.toString();
            });

            process.on('close', (code: number) => {
                if (code === 0) {
                    resolve();
                } else {
                    reject(new Error(stderr || `Export process exited with code ${code}`));
                }
            });

            process.on('error', (error: Error) => {
                reject(error);
            });
        });
    }

    private async compilePDF(texPath: string): Promise<void> {
        const workspaceFolder = vscode.workspace.getWorkspaceFolder(vscode.Uri.file(texPath));
        const workingDir = workspaceFolder?.uri.fsPath || path.dirname(texPath);

        vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "Compiling PDF...",
            cancellable: false
        }, async (progress) => {
            return new Promise<void>((resolve, reject) => {
                const { spawn } = require('child_process');
                const process = spawn('pdflatex', ['-interaction=nonstopmode', texPath], {
                    cwd: workingDir
                });

                let stderr = '';

                process.stderr.on('data', (data: Buffer) => {
                    stderr += data.toString();
                });

                process.on('close', (code: number) => {
                    if (code === 0) {
                        const pdfPath = texPath.replace('.tex', '.pdf');
                        vscode.window.showInformationMessage(
                            'PDF compiled successfully!',
                            'Open PDF'
                        ).then(selection => {
                            if (selection === 'Open PDF') {
                                vscode.env.openExternal(vscode.Uri.file(pdfPath));
                            }
                        });
                        resolve();
                    } else {
                        vscode.window.showErrorMessage(
                            'PDF compilation failed. Make sure pdflatex is installed.',
                            'Show Log'
                        ).then(selection => {
                            if (selection === 'Show Log') {
                                vscode.window.showInformationMessage(stderr);
                            }
                        });
                        reject(new Error('PDF compilation failed'));
                    }
                });

                process.on('error', (error: Error) => {
                    vscode.window.showErrorMessage(`Failed to start pdflatex: ${error.message}`);
                    reject(error);
                });
            });
        });
    }
}