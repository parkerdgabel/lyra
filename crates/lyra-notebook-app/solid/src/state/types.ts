// TS mirrors of core schema (UUID as string)

export type Assoc = Record<string, any>;

export type CellType = 'Code' | 'Markdown' | 'Text' | 'Output' | 'Graphics' | 'Table' | 'Raw';

export interface DisplayData {
  mime: string;
  data: string;
  schema?: Assoc;
  meta?: Assoc;
}

export interface CellMeta extends Assoc {
  execCount?: number;
  timingMs?: number;
  cached?: boolean;
  error?: string;
}

export interface Cell {
  id: string;
  type: CellType;
  language: string;
  attrs: number; // bitflags
  labels: string[];
  tags: string[];
  input: string;
  output: DisplayData[];
  meta: CellMeta;
}

export interface Notebook {
  id: string;
  version: string;
  metadata: Assoc;
  cells: Cell[];
  styles: Assoc;
  resources: Assoc;
}

// Kernel types
export interface ExecResult {
  cell_id: string;
  duration_ms: number;
  outputs: DisplayData[];
  error?: string | null;
}

export type ExecEvent =
  | { Started: { cell_id: string } }
  | { Output: { cell_id: string; item: DisplayData } }
  | { Finished: { result: ExecResult } }
  | { Error: { cell_id: string; message: string } };

