import type { ActionStatus, InferenceAdapter } from '../types/pinn';
import { appConfig } from './config';
import { createDemoAdapter } from './demoPhysics';
import { createRemoteAdapter } from './remoteAdapter';
import { formatTimestamp } from './utils';

interface CreateInferenceAdapterOptions {
  updateStatus: (label: keyof Record<string, unknown>, status: ActionStatus) => void;
}

function mapLabel(label: string): keyof Record<string, unknown> {
  if (label === 'query') {
    return 'probe';
  }
  return label;
}

export function createInferenceAdapter(
  options: CreateInferenceAdapterOptions
): InferenceAdapter {
  const publish = (
    label: string,
    state: ActionStatus['state'],
    detail: string
  ) => {
    options.updateStatus(mapLabel(label), {
      label,
      state,
      detail,
      updatedAt: formatTimestamp()
    });
  };

  if (appConfig.inferenceMode === 'remote') {
    return createRemoteAdapter({
      onProgress(event) {
        publish(event.label, event.state, event.message);
      }
    });
  }

  return createDemoAdapter((label, state) => {
    publish(label, state, state === 'running' ? '本地演示数值计算中。' : '本地演示数值计算完成。');
  });
}
