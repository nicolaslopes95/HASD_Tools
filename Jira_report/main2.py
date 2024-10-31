import logging
from jira_connection import get_jira_connection
from html_report import generate_html_report
import os
from utilities import send_html_email
from datetime import datetime, timedelta
import pandas as pd
import json
import sys
from dotenv import load_dotenv  # Ajouter cette ligne
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()



# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_worklogs_to_json(df):
    """
    Converts the worklogs DataFrame into a nested JSON structure organized by Epic, Issue, and Worklogs.

    :param df: pandas DataFrame containing worklog information
    :return: JSON string with the desired structure
    """
    json_data = {}

    # Group by Epic Key
    grouped_epics = df.groupby(['Epic Key', 'Epic Name', 'Epic Labels'])

    for (epic_key, epic_name, epic_labels), epic_group in grouped_epics:
        # Initialize Epic entry
        if epic_key not in json_data:
            json_data[epic_key] = {
                'Epic Name': epic_name,
                'Epic Labels': [label.strip() for label in epic_labels.split(',')] if isinstance(epic_labels, str) else [],
                'Issues': {}
            }

        # Group by Issue within the Epic
        grouped_issues = epic_group.groupby(['Issue Key', 'Summary', 'Issue Type'])

        for (issue_key, summary, issue_type), issue_group in grouped_issues:
            # Initialize Issue entry
            if issue_key not in json_data[epic_key]['Issues']:
                json_data[epic_key]['Issues'][issue_key] = {
                    'Summary': summary,
                    'Issue Type': issue_type,
                    'Worklogs': []
                }

            # Add Worklogs to the Issue
            for _, row in issue_group.iterrows():
                worklog = {
                    'Worklog Author': row['Worklog Author'],
                    'Time Spent': row['Time Spent'],
                    'Worklog Created': row['Worklog Created'],
                    'Worklog Comment': row['Worklog Comment']
                }
                json_data[epic_key]['Issues'][issue_key]['Worklogs'].append(worklog)

    # Convert the dictionary to a JSON string with indentation for readability
    return json.dumps(json_data, indent=4)

def extract_worklogs_in_period(jira, project, start_date, end_date, epic_field_id):
    """
    Extracts worklogs entered in a JIRA project within a given period.

    :param jira: Connected JIRA instance
    :param project: JIRA project key (e.g., 'PROJ')
    :param start_date: Start date in 'YYYY-MM-DD' format
    :param end_date: End date in 'YYYY-MM-DD' format
    :param epic_field_id: Custom field ID for Epic Link
    :return: DataFrame containing the extracted worklog information
    """

    # Convert dates to datetime objects
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    # JQL to search for issues in the project that have worklogs
    jql = f'''project = {project} AND (
        worklogDate >= "{start_date}" AND worklogDate <= "{end_date}"
        OR (
            status changed to Done 
            DURING ("{start_date}", "{end_date}")
        )
    )'''

    # Fields to retrieve
    fields = 'key,summary,issuetype,{epic_field_id},labels,parent'.format(epic_field_id=epic_field_id)

    try:
        # Search for issues
        issues = jira.search_issues(jql, fields=fields, expand='worklog', maxResults=False)
        logging.info(f"Retrieved {len(issues)} issues from JIRA.")
    except Exception as e:
        logging.error(f"Error searching issues: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

    # List to store data
    data = []
    epic_cache = {}

    for issue in issues:
        issue_type = issue.fields.issuetype.name
        epic_key = None
        epic_name = ''
        epic_labels = []
        summary = issue.fields.summary

        # Si c'est un Epic
        if issue_type == 'Epic':
            epic_key = issue.key
            epic_name = issue.fields.summary
            epic_labels = issue.fields.labels if hasattr(issue.fields, 'labels') else []
        
        # Si c'est une sous-tâche
        elif getattr(issue.fields.issuetype, 'subtask', False):
            try:
                # Vérifier si la tâche parente existe
                parent = getattr(issue.fields, 'parent', None)
                if parent is None:
                    logging.warning(f"Subtask {issue.key} has no parent task")
                    continue

                parent_key = parent.key
                parent_issue = jira.issue(parent_key)
                
                # Combiner le nom de la tâche mère avec la sous-tâche
                summary = f"{parent_issue.fields.summary} --> {issue.fields.summary}"
                
                # Récupérer l'Epic de la tâche parente
                parent_epic_key = getattr(parent_issue.fields, epic_field_id, None)
                
                if parent_epic_key:
                    if parent_epic_key in epic_cache:
                        epic_details = epic_cache[parent_epic_key]
                        epic_key = parent_epic_key
                        epic_name = epic_details['Epic Name']
                        epic_labels = epic_details['Epic Labels']
                    else:
                        epic_issue = jira.issue(parent_epic_key)
                        epic_key = parent_epic_key
                        epic_name = epic_issue.fields.summary
                        epic_labels = epic_issue.fields.labels
                        epic_cache[parent_epic_key] = {
                            'Epic Name': epic_name,
                            'Epic Labels': epic_labels
                        }
                else:
                    logging.warning(f"Parent task {parent_key} of subtask {issue.key} has no epic")
            except Exception as e:
                logging.error(f"Error retrieving epic details for subtask {issue.key}: {str(e)}")
        
        # Pour tous les autres types de tâches
        else:
            epic_key = getattr(issue.fields, epic_field_id, None)
            if epic_key:
                if epic_key in epic_cache:
                    epic_details = epic_cache[epic_key]
                    epic_name = epic_details['Epic Name']
                    epic_labels = epic_details['Epic Labels']
                else:
                    try:
                        epic_issue = jira.issue(epic_key)
                        epic_name = epic_issue.fields.summary
                        epic_labels = epic_issue.fields.labels
                        epic_cache[epic_key] = {
                            'Epic Name': epic_name,
                            'Epic Labels': epic_labels
                        }
                    except Exception as e:
                        logging.error(f"Error retrieving epic {epic_key}: {e}")
                        epic_cache[epic_key] = {'Epic Name': '', 'Epic Labels': []}

        # Retrieve work logs
        try:
            worklogs = jira.worklogs(issue.key)
        except Exception as e:
            logging.error(f"Error retrieving worklogs for issue {issue.key}: {e}")
            continue  # Skip to the next issue in case of error

        # Vérifier si la tâche a été terminée pendant la période
        status_changes = jira.issue(issue.key, expand='changelog').changelog.histories
        done_during_period = False
        done_by = None
        
        for history in status_changes:
            change_date = datetime.strptime(history.created[:10], '%Y-%m-%d')
            if start_dt <= change_date <= end_dt:
                for item in history.items:
                    if item.field == 'status' and item.toString == 'Done':
                        done_during_period = True
                        # Récupérer l'auteur du changement
                        author_name = history.author.displayName
                        done_by = ''.join(word[0].upper() for word in author_name.split())
                        break
                if done_during_period:
                    break

        # Si la tâche n'a pas de worklogs mais a été terminée pendant la période
        if done_during_period and not worklogs:
            data.append({
                'Issue Key': issue.key,
                'Summary': summary,
                'Issue Type': issue.fields.issuetype.name,
                'Epic Key': epic_key,
                'Epic Name': epic_name,
                'Epic Labels': ', '.join(epic_labels) if epic_labels else '',
                'Worklog Author': done_by,  # Utiliser l'auteur du changement
                'Time Spent': '0h',
                'Worklog Created': '',
                'Worklog Comment': '[Completed during this period]'
            })

        # Continuer avec le traitement existant des worklogs
        for worklog in worklogs:
            try:
                # Convert worklog.created to datetime
                worklog_created = datetime.strptime(worklog.created[:10], '%Y-%m-%d')
                if start_dt <= worklog_created <= end_dt:
                    # Get worklog comment if exists
                    worklog_comment = getattr(worklog, 'comment', '') or ''

                    data.append({
                        'Issue Key': issue.key,
                        'Summary': summary,
                        'Issue Type': issue.fields.issuetype.name,
                        'Epic Key': epic_key,
                        'Epic Name': epic_name,
                        'Epic Labels': ', '.join(epic_labels) if epic_labels else '',
                        'Worklog Author': worklog.author.displayName,
                        'Time Spent': worklog.timeSpent,
                        'Worklog Created': worklog.created,
                        'Worklog Comment': worklog_comment
                    })
            except AttributeError as ae:
                logging.error(f"Attribute error processing worklog for issue {issue.key}: {ae}")
            except Exception as e:
                logging.error(f"Error processing worklog for issue {issue.key}: {e}")

    # Create a DataFrame
    df = pd.DataFrame(data)

    logging.info(f"Extracted {len(df)} worklogs from JIRA.")
    return df





def calculate_time_per_person_and_label(worklogs_df):
    """
    Calculates the time spent by each person on each Epic label.

    :param worklogs_df: DataFrame containing worklog data
    :return: DataFrame with time spent per person and per Epic label
    """
    # Convert 'Time Spent' to hours
    def time_to_hours(time_str):
        if time_str.endswith('d'):
            return float(time_str[:-1]) * 8  # 1 day = 8 hours
        elif time_str.endswith('h'):
            return float(time_str[:-1])
        elif time_str.endswith('m'):
            return float(time_str[:-1]) / 60
        elif time_str.endswith('s'):
            return float(time_str[:-1]) / 3600
        else:
            return 0

    worklogs_df['Hours Spent'] = worklogs_df['Time Spent'].apply(time_to_hours)

    # Split Epic labels into individual rows
    worklogs_df['Epic Label'] = worklogs_df['Epic Labels'].str.split(',')
    worklogs_df = worklogs_df.explode('Epic Label').reset_index(drop=True)
    worklogs_df['Epic Label'] = worklogs_df['Epic Label'].str.strip()

    # Group by author and Epic label, then sum the time spent
    time_per_person_and_label = worklogs_df.groupby(['Worklog Author', 'Epic Label'])['Hours Spent'].sum().reset_index()

    # Round to two decimal places
    time_per_person_and_label['Hours Spent'] = time_per_person_and_label['Hours Spent'].round(2)

    return time_per_person_and_label

def calculate_time_per_person_and_epic(worklogs_df):
    """
    Calculates the time spent by each person on each Epic.

    :param worklogs_df: DataFrame containing worklog data
    :return: DataFrame with time spent per person and per Epic
    """
    # Convert 'Time Spent' to hours
    def time_to_hours(time_str):
        if time_str.endswith('d'):
            return float(time_str[:-1]) * 8  # 1 day = 8 hours
        elif time_str.endswith('h'):
            return float(time_str[:-1])
        elif time_str.endswith('m'):
            return float(time_str[:-1]) / 60
        elif time_str.endswith('s'):
            return float(time_str[:-1]) / 3600
        else:
            return 0

    worklogs_df['Hours Spent'] = worklogs_df['Time Spent'].apply(time_to_hours)

    # Group by author and Epic, then sum the time spent
    time_per_person_and_epic = worklogs_df.groupby(['Worklog Author', 'Epic Name'])['Hours Spent'].sum().reset_index()

    # Round to two decimal places
    time_per_person_and_epic['Hours Spent'] = time_per_person_and_epic['Hours Spent'].round(2)

    return time_per_person_and_epic

def filter_time_by_label(total_time_per_person_and_label, labels_to_exclude):
    """
    Filters the total_time_per_person_and_label DataFrame by excluding specified labels
    and also returns a DataFrame with only the specified labels.

    :param total_time_per_person_and_label: DataFrame containing time spent per person and per label
    :param labels_to_exclude: List of labels to exclude
    :return: Tuple containing two DataFrames:
             1. Filtered DataFrame without the specified labels
             2. DataFrame containing only the specified labels
    """
    # Create boolean masks for rows to keep and exclude
    mask_exclude = ~total_time_per_person_and_label['Epic Label'].isin(labels_to_exclude)
    mask_include = total_time_per_person_and_label['Epic Label'].isin(labels_to_exclude)
    
    # Apply the masks to filter the DataFrame
    filtered_df_exclude = total_time_per_person_and_label[mask_exclude].copy()
    filtered_df_include = total_time_per_person_and_label[mask_include].copy()
    
    # Reset the index of the filtered DataFrames
    filtered_df_exclude.reset_index(drop=True, inplace=True)
    filtered_df_include.reset_index(drop=True, inplace=True)
    
    return filtered_df_exclude, filtered_df_include

def create_graph_from_filtered_df(filtered_df_exclude):
    """
    Creates a pie chart from the filtered DataFrame.

    :param filtered_df_exclude: DataFrame containing filtered data
    :return: None (displays the graph)
    """
    import matplotlib.pyplot as plt

    # Calculate total hours per Epic label
    total_hours_per_label = filtered_df_exclude.groupby('Epic Label')['Hours Spent'].sum()

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a pie chart
    ax.pie(total_hours_per_label.values, labels=total_hours_per_label.index, autopct='%1.1f%%', startangle=90)

    # Customize the graph
    plt.title("Time Distribution by Epic Label")
    plt.axis('equal')  # Ensures the pie chart is circular

    # Display the graph
    plt.show()

    # Optionally, save the graph
    # plt.savefig('graph_filtered_df_pie.png')
def create_csv_from_json(worklogs_json):
    """
    Crée une chaîne CSV à partir des données JSON des journaux de travail.

    :param worklogs_json: Chaîne JSON contenant les données des journaux de travail
    :return: Chaîne CSV contenant les détails des journaux de travail
    """
    import json
    import csv
    from io import StringIO

    data = json.loads(worklogs_json)
    csv_output = StringIO()
    csv_writer = csv.writer(csv_output)

    csv_writer.writerow(['Epic', 'Task Key', 'Task', 'Task Type', 'Worklog Author', 'Comment'])

    previous_row = None
    for epic_key, epic_data in data.items():
        epic_name = epic_data['Epic Name']
        for issue_key, issue_data in epic_data['Issues'].items():
            issue_summary = issue_data['Summary']
            issue_type = issue_data['Issue Type']
            for worklog in issue_data['Worklogs']:
                current_row = [
                    epic_name,
                    issue_key,
                    issue_summary,
                    issue_type,
                    worklog['Worklog Author'],
                    worklog['Worklog Comment']
                ]
                if current_row != previous_row:
                    if current_row[5] or (previous_row is None or current_row[:5] != previous_row[:5]):
                        csv_writer.writerow(current_row)
                        previous_row = current_row

    return csv_output.getvalue()

# Exemple d'utilisation dans la fonction principale :
# csv_output = create_csv_from_json(worklogs_json)
# print(csv_output)  # Ou utilisez la chaîne CSV selon vos besoins

def create_html_table_from_json(worklogs_json, filtered_df_exclude, filtered_df_include, planned_tasks_df):
    """
    Creates an HTML table from worklog JSON data and planned tasks.
    """
    import json
    import matplotlib.pyplot as plt
    import io
    import base64

    # Fonction pour créer un graphique en camembert et le convertir en base64
    def create_pie_chart(df, title,type_colour):
        plt.figure(figsize=(8, 6))
        total_hours_per_label = df.groupby('Epic Label')['Hours Spent'].sum()
        
        # Palette de couleurs pour les labels non standards (graphique de gauche)
        other_colors = [
            '#FF9999',  # Rose clair
            '#66B2FF',  # Bleu clair
            '#99FF99',  # Vert clair
            '#FFCC99',  # Orange clair
            '#FF99CC',  # Rose foncé
            '#99CCFF',  # Bleu pastel
            '#FFB366',  # Orange foncé
            '#99FF99',  # Vert pastel
            '#FF99FF',  # Violet clair
            '#99FFCC',  # Turquoise
        ]
        
        # Définir les couleurs correspondant aux labels standards
        standard_color_mapping = {
            'B2C': '#3498db',      # Bleu
            'B2B': '#e74c3c',      # Rouge
            'Licensing': '#2ecc71', # Vert
            'RI': '#f1c40f',       # Jaune
            'No Label': '#95a5a6'  # Gris
        }
        
        # Créer la liste des couleurs en fonction du titre du graphique
        if type_colour == 1:  # Graphique de droite
            colors = [standard_color_mapping.get(label, '#95a5a6') for label in total_hours_per_label.index]
        else:  # Graphique de gauche
            # Trier les labels par ordre alphabétique et attribuer les couleurs
            sorted_labels = sorted(total_hours_per_label.index)
            color_dict = dict(zip(sorted_labels, other_colors[:len(sorted_labels)]))
            colors = [color_dict[label] for label in total_hours_per_label.index]
        
        plt.pie(total_hours_per_label.values, 
                labels=total_hours_per_label.index, 
                autopct='%1.1f%%', 
                colors=colors)
        
        plt.title(title)
        plt.axis('equal')
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
        
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        return base64.b64encode(image_png).decode()

    # Créer les deux graphiques
    chart1 = create_pie_chart(filtered_df_exclude, "Distribution du temps par projet",0)
    chart2 = create_pie_chart(filtered_df_include, "Distribution du temps",1)

    data = json.loads(worklogs_json)
    
    # Créer une structure unifiée pour les worklogs et les tâches planifiées
    unified_data = {
        'B2C': {},
        'B2B': {},
        'Licensing': {},
        'RI': {},
        'No Label': {}
    }

    # D'abord, créer un dictionnaire des epics depuis worklogs_json pour avoir les labels corrects
    epic_label_mapping = {}  # Pour stocker {epic_key: label}
    
    # Ajouter les worklogs à la structure unifiée et construire le mapping des epics
    for epic_key, epic_data in data.items():
        epic_labels = epic_data.get('Epic Labels', [])
        assigned_label = 'No Label'
        for label in ['B2C', 'B2B', 'Licensing', 'RI']:
            if label in epic_labels:
                assigned_label = label
                break
        
        epic_label_mapping[epic_key] = assigned_label
        
        if epic_key not in unified_data[assigned_label]:
            unified_data[assigned_label][epic_key] = {
                'epic_name': epic_data['Epic Name'],
                'worklogs': epic_data['Issues'],
                'planned_tasks': []
            }

    # Ajouter les tâches planifiées en utilisant le mapping des epics existant
    for _, task in planned_tasks_df.iterrows():
        epic_info = task['Epic'].split(' - ', 1) if ' - ' in task['Epic'] else ['No Epic', task['Epic']]
        epic_key = epic_info[0]
        
        # Si l'epic existe déjà dans notre mapping, utiliser son label
        if epic_key in epic_label_mapping:
            assigned_label = epic_label_mapping[epic_key]
        else:
            # Sinon, déterminer le label à partir du nom de l'epic ou du résumé
            epic_name = epic_info[1]
            assigned_label = 'No Label'
            for label in ['B2C', 'B2B', 'Licensing', 'RI']:
                if label.lower() in epic_name.lower() or label.lower() in task['Summary'].lower():
                    assigned_label = label
                    break
        
        # Si l'epic n'existe pas encore dans la structure, l'ajouter
        if epic_key not in unified_data[assigned_label]:
            unified_data[assigned_label][epic_key] = {
                'epic_name': epic_info[1],
                'worklogs': {},
                'planned_tasks': []
            }
        
        # Ajouter la tâche planifiée à l'epic
        unified_data[assigned_label][epic_key]['planned_tasks'].append(task)

    # Générer le HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body {{ font-family: Arial, sans-serif; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        ul {{ list-style-type: none; padding-left: 0; }}
        .task {{ margin-left: 20px; list-style-type: circle; }}
        .subtask {{ margin-left: 40px; list-style-type: square; }}
        .worklog {{ margin-left: 20px; font-style: italic; color: #666; }}
        .author {{ color: #888; font-size: 0.9em; }}
        .label-cell {{
            width: 100px;
            text-align: center;
            font-weight: bold;
            color: white;
        }}
        .label-B2C {{ background-color: #3498db; }}
        .label-B2B {{ background-color: #e74c3c; }}
        .label-Licensing {{ background-color: #2ecc71; }}
        .label-RI {{ background-color: #f1c40f; }}
        .label-NoLabel {{ background-color: #95a5a6; }}
        .charts-container {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
        }}
        .chart {{
            text-align: center;
        }}
        .chart img {{
            max-width: 100%;
            height: auto;
        }}
        .planned-tasks-header {{
            margin-top: 10px;
            font-weight: bold;
            color: #666;
            border-bottom: 1px solid #ddd;
        }}
        .task.planned {{
            background-color: #f8f9fa;
            //border-left: 3px solid #007bff;
            //padding-left: 5px;
            //margin: 5px 0;
        }}
    </style>
    </head>
    <body>
    <div class="charts-container">
        <div class="chart">
            <img src="data:image/png;base64,{chart1}" alt="Distribution hors B2C/B2B/RI">
        </div>
        <div class="chart">
            <img src="data:image/png;base64,{chart2}" alt="Distribution B2C/B2B/RI">
        </div>
    </div>
    <table>
    """

    for label, epics in unified_data.items():
        if epics:  # Si le label contient des epics
            label_class = f"label-{label.replace(' ', '')}"
            
            # Compter le nombre total de lignes pour ce label
            total_rows = len(epics)
            
            first_epic = True
            for epic_key, epic_data in epics.items():
                if not first_epic:
                    html += "<tr>"
                else:
                    html += f"""
                    <tr>
                        <td class="label-cell {label_class}" rowspan="{total_rows}">{label}</td>
                    """
                
                epic_name = epic_data['epic_name']
                details = "<ul>"
                
                # Ajouter l'en-tête "Progress :"
                details += '<li class="planned-tasks-header">Progress :</li>'
                
                # Ajouter les worklogs
                for issue_key, issue_data in epic_data['worklogs'].items():
                    issue_summary = issue_data['Summary']
                    details += f'<li class="task"><a href="https://devialet.atlassian.net/browse/{issue_key}">{issue_key}</a>: {issue_summary}'
                    
                    # Group worklogs by whether they have comments
                    worklogs_with_comments = []
                    authors_without_comments = set()
                    
                    for worklog in issue_data['Worklogs']:
                        if worklog['Worklog Comment']:
                            worklogs_with_comments.append(worklog)
                        else:
                            author = worklog['Worklog Author']
                            initials = ''.join(word[0].upper() for word in author.split())
                            authors_without_comments.add(initials)
                    
                    if authors_without_comments:
                        details += f' <span class="author">[{", ".join(sorted(authors_without_comments))}]</span>'
                    
                    for worklog in worklogs_with_comments:
                        author = worklog['Worklog Author']
                        initials = ''.join(word[0].upper() for word in author.split())
                        details += f'<div class="worklog">{worklog["Worklog Comment"]} [{initials}]</div>'
                    
                    details += '</li>'
                
                # Ajouter les tâches planifiées
                if epic_data['planned_tasks']:
                    # Modifier "Planned Tasks:" en "Plan :"
                    details += '<li class="planned-tasks-header">Plan :</li>'
                    for task in epic_data['planned_tasks']:
                        assignee_initials = ''.join(word[0].upper() for word in task['Assignee'].split()) if task['Assignee'] != 'Unassigned' else 'UA'
                        details += f'''
                            <li class="task planned">
                                <a href="https://devialet.atlassian.net/browse/{task['Key']}">{task['Key']}</a>: 
                                {task['Summary']} 
                                <span class="author">[{assignee_initials}]</span>
                                <span style="color: #666;">({task['Status']})</span>
                            </li>
                        '''
                
                details += "</ul>"
                
                html += f"""
                    <td>{epic_name}</td>
                    <td>{details}</td>
                </tr>
                """
                first_epic = False

    html += """
    </table>
    </body>
    </html>
    """

    return html

def get_planned_tasks(jira, project):
    """
    Récupère toutes les tâches, sous-tâches et stories en To Do ou In Progress.
    
    :param jira: Instance JIRA connectée
    :param project: Clé du projet JIRA
    :return: DataFrame avec les tâches planifiées
    """
    # JQL pour récupérer les tâches planifiées
    jql_planned = f'''project = {project} 
        AND status in ("To Do", "In Progress") 
        AND issuetype in (Story, Sub-task, Task)
        ORDER BY status ASC, created DESC'''

    try:
        # Champs à récupérer
        fields = 'key,summary,status,issuetype,priority,assignee,customfield_10008'  # customfield_10008 est l'Epic Link
        
        # Récupérer les issues
        planned_issues = jira.search_issues(jql_planned, fields=fields, maxResults=False)
        logging.info(f"Retrieved {len(planned_issues)} planned issues")

        # Préparer les données pour le DataFrame
        planned_data = []
        for issue in planned_issues:
            # Récupérer l'epic
            epic_key = getattr(issue.fields, 'customfield_10008', None)
            epic_name = ''
            if epic_key:
                try:
                    epic = jira.issue(epic_key)
                    epic_name = epic.fields.summary
                except:
                    epic_name = 'Unknown Epic'

            # Récupérer l'assignee
            assignee = getattr(issue.fields.assignee, 'displayName', 'Unassigned') if issue.fields.assignee else 'Unassigned'

            planned_data.append({
                'Key': issue.key,
                'Type': issue.fields.issuetype.name,
                'Status': issue.fields.status.name,
                'Priority': issue.fields.priority.name,
                'Summary': issue.fields.summary,
                'Assignee': assignee,
                'Epic': f"{epic_key} - {epic_name}" if epic_key else 'No Epic'
            })

        # Créer le DataFrame
        df = pd.DataFrame(planned_data)
        
        # Afficher le tableau dans la console
        print("\n=== PLANNED TASKS ===")
        print(df.to_string(index=False))
        
        return df

    except Exception as e:
        logging.error(f"Error retrieving planned tasks: {e}")
        return pd.DataFrame()

def send_html_email(html_content, start_date, end_date):
    """
    Envoie le rapport HTML par email.
    """
    # Récupérer les informations d'email depuis les variables d'environnement
    smtp_server = os.getenv('SMTP_SERVER')
    smtp_port = int(os.getenv('SMTP_PORT', '587'))
    sender_email = os.getenv('SENDER_EMAIL')
    sender_password = os.getenv('SENDER_PASSWORD')
    recipient_emails = os.getenv('RECIPIENT_EMAILS').split(',')  # Liste d'emails séparés par des virgules

    # Créer le message
    msg = MIMEMultipart('alternative')
    msg['Subject'] = f'Rapport d\'activité ASD ({start_date} au {end_date})'
    msg['From'] = sender_email
    msg['To'] = ', '.join(recipient_emails)

    # Ajouter le contenu HTML
    msg.attach(MIMEText(html_content, 'html'))

    try:
        # Connexion au serveur SMTP
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        logging.info("Email envoyé avec succès")
        return True
    except Exception as e:
        logging.error(f"Erreur lors de l'envoi de l'email: {str(e)}")
        return False

def main():
    # Load environment variables
    jira_server = os.getenv('JIRA_SERVER')
    jira_email = os.getenv('JIRA_EMAIL')
    jira_token = os.getenv('JIRA_TOKEN')
    epic_field_id = os.getenv('JIRA_EPIC_FIELD_ID', 'customfield_10008')
    email_recipient = os.getenv('EMAIL_RECIPIENT')

    # Validate environment variables
    required_env_vars = ['JIRA_SERVER', 'JIRA_EMAIL', 'JIRA_TOKEN']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logging.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)

    try:
        logging.info("Connecting to JIRA...")
        jira = get_jira_connection(jira_server, jira_email, jira_token)
        logging.info("Successfully connected to JIRA.")

        project_key = 'ASD'  # Replace with your project key or make it configurable
        # Get dates from last Monday to this Friday
        today = datetime.now()
        start_date = (today - timedelta(days=today.weekday() + 7)).strftime('%Y-%m-%d')  # Last Monday
        end_date = (today + timedelta(days=4 - today.weekday())).strftime('%Y-%m-%d')    # This Friday

        # Récupérer les tâches planifiées
        planned_tasks_df = get_planned_tasks(jira, project_key)

        logging.info(f"Extracting worklogs for project {project_key} from {start_date} to {end_date}.")
        worklogs_df = extract_worklogs_in_period(jira, project_key, start_date, end_date, epic_field_id)
        total_time_per_person_and_label = calculate_time_per_person_and_label(worklogs_df)
        total_time_per_person_and_epic = calculate_time_per_person_and_epic(worklogs_df)
        filtered_df_exclude, filtered_df_include = filter_time_by_label(
            total_time_per_person_and_label, 
            ['B2C', 'B2B', 'RI', 'Licensing']
        )
        #create_graph_from_filtered_df(filtered_df_exclude)
        #create_graph_from_filtered_df(filtered_df_include)

        # Display the table
        print(worklogs_df)

        # Optionally: save to CSV
        csv_file = 'jira_worklogs.csv'
        worklogs_df.to_csv(csv_file, index=False)
        logging.info(f"Worklogs saved to {csv_file}.")

        if not worklogs_df.empty:
            # Convert DataFrame to JSON
            worklogs_json = convert_worklogs_to_json(worklogs_df)
            csv_output = create_csv_from_json(worklogs_json)
            
            # Calculer les données pour les graphiques
            total_time_per_person_and_label = calculate_time_per_person_and_label(worklogs_df)
            filtered_df_exclude, filtered_df_include = filter_time_by_label(
                total_time_per_person_and_label, 
                ['B2C', 'B2B', 'RI', 'Licensing']
            )
            
            # Créer le tableau HTML avec les graphiques et les tâches planifiées
            html_table = create_html_table_from_json(
                worklogs_json,
                filtered_df_exclude,
                filtered_df_include,
                planned_tasks_df
            )

            print(csv_output)
            # Save HTML to file
            html_file = 'worklogs_table.html'
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_table)
            logging.info(f"HTML table saved to {html_file}")
            # Print rendered HTML table to console
            from bs4 import BeautifulSoup
            
            # Parse and prettify the HTML for better console display
            soup = BeautifulSoup(html_table, 'html.parser')
            pretty_html = soup.prettify()
            
            print("\nRendered HTML Output:")
            print(pretty_html)

            # Display the JSON
            print(worklogs_json)

            # Optionally: save to a JSON file
            json_file = 'jira_worklogs.json'
            with open(json_file, 'w', encoding='utf-8') as f:
                f.write(worklogs_json)
            logging.info(f"Worklogs JSON saved to {json_file}.")

            # Envoyer l'email avec le rapport HTML
            email_sent = send_html_email(html_table, start_date, end_date)
            
            if not email_sent:
                logging.error("Échec de l'envoi de l'email")
                sys.exit(1)
        else:
            logging.info("No worklogs found for the specified period.")

    except Exception as e:
        logging.exception(f"Error connecting to JIRA or processing worklogs: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

