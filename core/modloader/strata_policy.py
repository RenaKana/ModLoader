"""Runtime-Python bridge for prepared resources and model identity.

The official service owns readiness/retry and hardware probes. Its drafter
discovery assumes setup's directory layout and uses the API alias as a model
family; ModLoader supplies those identities instead. Confidence and memory
planning remain the installed Control version's algorithms.
"""
import json
import inspect
import runpy
import sys


def normalized(args):
    return [part for arg in args for part in (arg.split('=', 1) if arg.startswith('--') and '=' in arg else [arg])]


def install():
    from strata_control import drafting, vram_fit, user_args
    original_apply, original_fit = drafting.apply, vram_fit.fit_start

    def apply(cfg, **hardware):
        policy = cfg.get('_modloader_policy')
        if policy is None:
            return original_apply(cfg, **hardware)
        asked = cfg.get('drafting', 'off')
        effective = policy['initial'] if asked == 'auto' else asked
        # The service's one readiness retry asks for MTP. Do not reacquire or
        # reselect DFlash2 when no prepared MTP exists: retry without a drafter.
        if effective == 'mtp' and not policy['mtp_path']:
            effective = 'off'
        args, mine = normalized(cfg['args']), normalized(cfg.get('user_args', []))
        args = user_args.merge(args, mine)
        base = drafting.strip_mtp_flags(drafting.strip_draft_flags(args))
        spec = drafting._flag_value(mine, '--spec')
        base = drafting._without_flag(base, '--spec') + ['--spec', spec or ('6' if effective == 'dflash2' else '4')]
        flags = []
        if effective == 'dflash2':
            flags = list(drafting.engine_flags(policy['path'], policy['vocab']))
            flags = drafting._without_flag(flags, '--spec')
            confidence = drafting._flag_value(mine, '--draft-conf')
            flags[flags.index('--draft-conf') + 1] = confidence if confidence is not None else f"{drafting.conf_for(hardware.get('backend'), drafting.slots_of(base, hardware.get('vram_mib'))):g}"
            for flag in drafting.DRAFT_FLAGS:
                if flag in {'--draft', '--draft-model', '--draft-conf', '--draft-fallback'}:
                    continue
                value = drafting._flag_value(mine, flag)
                if value is not None:
                    flags += [flag, value]
        elif effective == 'mtp':
            flags = ['--mtp', policy['mtp_path']]
            mtp_args = user_args.merge(policy.get('mtp_args', []), mine)
            for flag, takes_value in drafting.MTP_ONLY_FLAGS.items():
                if flag == '--mtp':
                    continue
                value = drafting._flag_value(mtp_args, flag)
                if takes_value and value is not None:
                    flags += [flag, value]
                elif not takes_value and flag in mtp_args:
                    flags.append(flag)
        cfg['args'] = base + flags
        cfg['mtp'] = effective == 'mtp'
        return drafting.Decision(asked, effective, flags=tuple(flags) if effective == 'dflash2' else ())

    def fit_start(cfg, args, vram_mib, backend):
        policy = cfg.get('_modloader_policy')
        projected = {**cfg, 'model_name': policy['model_name'], 'user_args': normalized(cfg.get('user_args', []))} if policy else cfg
        return original_fit(projected, normalized(args), vram_mib, backend)

    drafting.apply, vram_fit.fit_start = apply, fit_start


def main():
    root = sys.argv.pop(1)
    sys.path.insert(0, root)
    if sys.argv[1:2] == ['--plan']:
        from contextlib import redirect_stdout
        cfg = json.load(sys.stdin)
        if '_modloader_policy' in cfg:
            install()
        from serve import server
        parameters = inspect.signature(server.engine_args).parameters
        kwargs = {}
        if 'exe' in parameters:
            kwargs['exe'] = cfg.get('exe')
        if 'announce' in parameters:
            kwargs['announce'] = False
        with redirect_stdout(sys.stderr):
            args = server.engine_args(cfg, **kwargs)
        print(json.dumps({'args': args}))
    else:
        install()
        runpy.run_module('serve.server', run_name='__main__')


if __name__ == '__main__':
    main()
